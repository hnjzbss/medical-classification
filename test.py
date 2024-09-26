import os
import argparse
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import resnet34
import torch.nn as nn
from mimic_dataset import MIMICCXRDataset
from data_tools import evaluate
from data_config import ClassificationConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args, config):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    test_df_path = os.path.join(args.csv_path, "test_label.csv")
    test_df = pd.read_csv(test_df_path)

    test_data_set = MIMICCXRDataset(test_df, data_path=args.data_path, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_data_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw, drop_last=False)

    model = resnet34()
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, args.num_classes)
    model.to(device)

    # load model weights
    weights_path = config.save_weight_name
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)

    logs_info = []
    test_mean_auc_list = []
    with torch.no_grad():
        auroc, mean_auroc, f1, mean_f1 = evaluate(model=model,
                                                  data_loader=test_loader,
                                                  device=device)

        data = [[*auroc, mean_auroc], [*f1, mean_f1]]
        columns = [*args.label_list, 'Mean']
        index = ['AUROC', 'F1 Score']
        df = pd.DataFrame(data=data, index=index, columns=columns)
        if args.display_f1 is True:
            print(df.transpose())
        else:
            print('Values for AUROC:')
            print(df.loc['AUROC'].to_string())

        logs_info.append(df.transpose())
        config.save_test_log(test_log_name=config.test_log_name, logs_info=logs_info)


if __name__ == '__main__':
    config = ClassificationConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=config.num_classes)
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--batch-size', type=int, default=config.batch_size)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    parser.add_argument('--data-path', type=str,
                        default=config.data_path)

    parser.add_argument('--csv-path', type=str,
                        default=config.csv_path)
    parser.add_argument('--debug', type=bool,
                        default=config.debug)
    parser.add_argument('--display_f1', type=bool,
                        default=config.display_f1)
    parser.add_argument('--label_list', type=str,
                        default=config.label_list)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt, config)
