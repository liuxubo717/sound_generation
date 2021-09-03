import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from apex import amp

except ImportError:
    amp = None

from datasets import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler
from datasets import CodeRow

def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (bottom, class_id, salience, file_name) in enumerate(loader):
        model.zero_grad()

        class_id = torch.FloatTensor(list(map(eval, list(class_id)))).long().unsqueeze(1)
        # salience = torch.FloatTensor(list(map(eval, list(salience)))).unsqueeze(1)

        bottom = bottom.to(device)
        class_id = class_id.to(device)
        # salience = salience.to(device)

        target = bottom
        # print(target[0])
        out, _ = model(bottom, label_condition=class_id)
        # out, _ = model(bottom, label_condition=class_id, salience_condition=salience)

        loss = criterion(out, target)
        # print(loss)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )





class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--hier', type=str, default='bottom')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--path', type=str, default='vqvae-code/')

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}
    start_point = 0

    if args.ckpt is not None:
        _, start_point = args.ckpt.split('_')
        start_point = int(start_point[0:-3])

        ckpt = torch.load(args.ckpt)
        args = ckpt['args']

    if args.hier == 'top':
        # model = PixelSNAIL(
        #     [10, 43],
        #     512,
        #     args.channel,
        #     5,
        #     4,
        #     args.n_res_block,
        #     args.n_res_channel,
        #     dropout=args.dropout,
        #     n_out_res_block=args.n_out_res_block,
        # )

        model = PixelSNAIL(
            [10, 43],
            512,
            256,
            5,
            4,
            4,
            256,
            dropout=0.1,
            n_out_res_block=0,
            cond_res_channel=0
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [20, 86],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            # attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(start_point, start_point + args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        torch.save(
                    {'model': model.module.state_dict(), 'args': args},
                    f'checkpoint/pixelsnail-final/{args.hier}_{str(i + 1).zfill(3)}.pt',
                )



