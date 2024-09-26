import os
import sys
import numpy as np
import metrics
import torch
from tqdm import tqdm
import json
import math


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function=None):
    model.train()
    if loss_function is None:
        loss_function = torch.nn.BCELoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


def evaluate(model, data_loader, device, return_result=False):
    model.eval()
    all_val_label = torch.FloatTensor().to(device)
    all_val_predict = torch.FloatTensor().to(device)
    with torch.no_grad():
        val_bar = tqdm(data_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))

            all_val_label = torch.cat((all_val_label, val_labels.to(device)), dim=0)
            all_val_predict = torch.cat((all_val_predict, outputs.to(device)), dim=0)

        all_val_label = all_val_label.cpu().numpy()
        all_val_predict = all_val_predict.cpu().numpy()

        thresholds = metrics.get_thresholds(all_val_label, all_val_predict)
        auroc = metrics.calculate_auroc(all_val_label, all_val_predict)
        thresholded_predictions = 1 * (all_val_predict > thresholds)
        f1 = metrics.calculate_f1(
            labels=all_val_label, predictions=thresholded_predictions)

        mean_auroc = np.mean(auroc)
        mean_f1 = np.mean(f1)

        if return_result:
            return auroc, mean_auroc, f1, mean_f1, all_val_label, all_val_predict
        else:
            return auroc, mean_auroc, f1, mean_f1
