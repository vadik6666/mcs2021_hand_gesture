import torch
import numpy as np
from tqdm import tqdm

from utils import AverageMeter
import utils
import os.path as osp
import os
from collections import OrderedDict


# Mixup (https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py#L119)
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    # print(alpha, lam)
    batch_size = x.shape[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          config, epoch, writer) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch (int): epoch number
    :return: None
    """
    # model.train()

    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')

    train_iter = tqdm(train_loader, desc='Train', dynamic_ncols=True, position=1)

    for step, (x, y) in enumerate(train_iter):
        lr = optimizer.param_groups[0]["lr"]

        x = x.cuda().to(memory_format=torch.contiguous_format)
        y = y.cuda()

        ####
        if hasattr(config.dataset, 'mixup_alpha'):
            mixup_alpha = config.dataset.mixup_alpha
        else:
            mixup_alpha = 0
        x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
        ####

        out = model(x)

        # loss = criterion(out, y)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        
        num_of_samples = x.shape[0]
        loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores = torch.softmax(out, dim=1).detach().cpu().numpy()
        predict = np.argmax(scores, axis=1)
        
        # gt = y.detach().cpu().numpy()
        gt_a = y_a.detach().cpu().numpy()
        gt_b = y_b.detach().cpu().numpy()

        # acc = np.mean(gt == predict)
        acc = lam * np.mean(gt_a == predict) + (1 - lam) * np.mean(gt_b == predict)

        acc_stat.update(acc, num_of_samples)

        if step % config.train.freq_vis == 0 and not step == 0:
            acc_val, acc_avg = acc_stat()
            loss_val, loss_avg = loss_stat()
            writer.add_scalar("train/loss", loss_avg, epoch * len(train_loader) + step)
            writer.add_scalar("train/acc", acc_avg, epoch * len(train_loader) + step)
            writer.add_scalar("train/lr", float(lr), epoch * len(train_loader) + step)
            print('Epoch: {}; step: {}; loss: {:.5f}; acc: {:.5f}'.format(epoch, step, loss_avg, acc_avg))

        if step % 500 == 0 and not step == 0:
            filename = "model_{:04d}_{}.pth".format(epoch,step)
            directory = osp.join(config.outdir, config.exp_name)
            filename = os.path.join(directory, filename)
            print('!!! Save intermidiate cp to ', filename)
            weights = model.state_dict()
            state = OrderedDict([
                ('state_dict', weights),
            ])

            torch.save(state, filename)



    acc_val, acc_avg = acc_stat()
    loss_val, loss_avg = loss_stat()
    print('Train process of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}'.format(epoch, loss_avg, acc_avg))


def validation(model: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               epoch, writer) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: None`
     """
    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')

    full_scores = []
    full_gt = []

    with torch.no_grad():
        # model.eval()
        val_iter = tqdm(val_loader, desc='Val', dynamic_ncols=True, position=2)

        for step, (x, y) in enumerate(val_iter):
            out = model(x.cuda().to(memory_format=torch.contiguous_format))
            loss = criterion(out, y.cuda())
            num_of_samples = x.shape[0]

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)

            scores = torch.softmax(out, dim=1).detach().cpu().numpy()

            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            # print(step, scores.shape, gt.shape)
            full_scores.append(scores)
            full_gt.append(gt)

            acc = np.mean(gt == predict)
            acc_stat.update(acc, num_of_samples)

        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        writer.add_scalar("val/loss", loss_avg, epoch)
        writer.add_scalar("val/acc", acc_avg, epoch)
        print('Validation of epoch: {} is done; \n loss: {:.5f}; acc: {:.5f}'.format(epoch, loss_avg, acc_avg))

    # ROC
    from sklearn.metrics import roc_curve
    # TARGET_FPR = 0.002
    full_scores = np.vstack(full_scores)
    full_gt = np.hstack(full_gt)
    print(full_scores.shape, full_gt.shape)

    for TARGET_FPR in [0.0001, 0.001, 0.002, 0.01, 0.05]:
        class_name2score_dict = {}

        for i in range(1, 7):
            # print(f'i={i}')
            y_pred = full_scores[:, i]
            fpr, tpr, thr = roc_curve(full_gt, y_pred, pos_label=i)

            if fpr[0] < TARGET_FPR:
                target_tpr = tpr[fpr < TARGET_FPR][-1]
            else:
                target_tpr = 0.0

            class_name2score_dict[i] = target_tpr

        mean_target_tpr = np.mean(list(class_name2score_dict.values()))
        writer.add_scalar(f"ROC/tpr_at_fpr={TARGET_FPR}", mean_target_tpr, epoch)
        print(f'TPR at FPR={TARGET_FPR}: Score: {mean_target_tpr}')