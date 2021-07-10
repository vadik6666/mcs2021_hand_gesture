import os
import sys
import yaml
import random
import argparse
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm

import utils
from models import models
from data import get_dataloader
from train import train, validation
from utils import convert_dict_to_tuple
from torch.utils.tensorboard import SummaryWriter


def main(args: argparse.Namespace) -> None:
    """
    Run train process of classification model
    :param args: all parameters necessary for launch
    :return: None
    """

    for i, x in enumerate(sys.argv):
        print(f'sys.argv[{i}]={x}')

    with open(args.cfg) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id

    outdir = osp.join(config.outdir, config.exp_name)
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # #---------- copy source code to outdir -----------------
    # if not os.path.exists(outdir+'/code'):
    #     os.makedirs(outdir+'/code')
    # if not os.path.exists(outdir+'/code/config'):
    #     os.makedirs(outdir+'/code/config')
    # if not os.path.exists(outdir+'/code/lists'):
    #     os.makedirs(outdir+'/code/lists')

    # os.system(f'rsync -avr *.py {outdir}/code/')
    # os.system(f'rsync -avr models {outdir}/code/')
    # os.system(f'rsync -avr data {outdir}/code/')
    # os.system(f'rsync -avr {sys.argv[2]} {outdir}/code/config/')
    # list_folder = '/'.join(config.dataset.train_annotation_main.split('/')[:-1])
    # os.system(f'rsync -avr {list_folder} {outdir}/code/lists/')
    # with open(f'{outdir}/code/execution_command.txt', 'w') as f:
    #     for i, x in enumerate(sys.argv):
    #         f.write(f'sys.argv[{i}]={x}\n')
    # #-------------------------------------------------------

    train_loader, val_loader = get_dataloader.get_dataloaders(config)

    print("Loading model...")
    net = models.load_model(config)
    print('net= ', net)
    print('\nconfig= ', config)
    print("Done.")

    criterion, criterion_val, optimizer, scheduler = utils.get_training_parameters(config, net)
    train_epoch = tqdm(range(config.train.n_epoch), dynamic_ncols=True, desc='Epochs', position=0)

    writer = SummaryWriter(f"{outdir}/tb")

    # main process
    for epoch in train_epoch:
        net.train()

        if hasattr(config.model, 'freeze_stages') and config.model.freeze_stages:
            print(f'Freezing first {config.model.freeze_stages} stages')
            stage2cnt = {1:5, 2:6, 3:7, 4:8}
            ct = 0
            for child in net.children():
                ct += 1
                if ct <= stage2cnt[config.model.freeze_stages]:
                    print(f'freezing layer[{ct}]: {child}')
                    child.eval()
                    for param in child.parameters():
                        param.requires_grad = False
        
        train(net, train_loader, criterion, optimizer, config, epoch, writer)

        net.eval()
        validation(net, val_loader, criterion_val, epoch, writer)

        utils.save_checkpoint(net, optimizer, scheduler, epoch, outdir)
        scheduler.step()

    # tensorboard off
    writer.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='Path to config file.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
