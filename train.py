import os
import math
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import resnet34
import torch.nn as nn
import pandas as pd
from data_tools import train_one_epoch, evaluate
from data_config import ClassificationConfig
from mimic_dataset import MIMICCXRDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main(args, config):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_df_path = "./data_file/train_label.csv"
    validate_df_path = "./data_file/validate_label.csv"
    train_df = pd.read_csv(train_df_path)
    validate_df = pd.read_csv(validate_df_path)

    if args.debug:
        train_df = train_df.sample(n=500)
        validate_df = validate_df.sample(n=500)

    train_data_set = MIMICCXRDataset(train_df, data_path=args.data_path, transform=data_transform["train"])
    val_data_set = MIMICCXRDataset(validate_df, data_path=args.data_path, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw, drop_last=True)

    net = resnet34()
    model_weight_path = args.weights
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, args.num_classes)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_auc = 0.0
    logs_info = []
    train_loss_list = []
    for epoch in range(args.epochs):
        mean_loss = train_one_epoch(model=net,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        scheduler.step()
        auroc, mean_auroc, f1, mean_f1 = evaluate(model=net,
                                                  data_loader=val_loader,
                                                  device=device)
        train_loss_list.append(mean_loss)
        print("*" * 30, "epoch: {:d}".format(epoch), "*" * 30)
        data = [[*auroc, mean_auroc], [*f1, mean_f1]]
        columns = [*args.label_list, 'Mean']
        index = ['AUROC', 'F1 Score']
        df = pd.DataFrame(data=data, index=index, columns=columns)
        if args.display_f1 is True:
            print(df.transpose())
        else:
            print('Values for AUROC:')
            print(df.loc['AUROC'].to_string())
        logs_info.append("*" * 30 + "epoch: {:d}".format(epoch) + "*" * 30)
        logs_info.append(df.transpose())
        config.save_train_log(config.train_log_name, args, logs_info)

        if mean_auroc > best_auc:
            best_auc = mean_auroc
            torch.save(net.state_dict(), config.save_weight_name)


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
    parser.add_argument('--weights', type=str, default='./resnet34-pre.pth',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt, config)
