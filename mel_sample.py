import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail import PixelSNAIL

import numpy as np


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, label_condition=None, salience_condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}


    label_condition = torch.full([batch, 1], label_condition).long().to(device)
    # salience_condition = torch.full([batch, 1], salience_condition).to(device)

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], label_condition=label_condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample
    # else:
    #     for i in tqdm(range(size[0])):
    #         for j in range(size[1]):
    #             out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
    #             prob = torch.softmax(out[:, :, i, j] / temperature, 1)
    #             sample = torch.multinomial(prob, 1).squeeze(-1)
    #             row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    # elif model == 'pixelsnail_top':
    #     model = PixelSNAIL(
    #         [10, 43],
    #         512,
    #         256,
    #         5,
    #         4,
    #         4,
    #         256,
    #         dropout=0.1,
    #         n_out_res_block=0,
    #         cond_res_channel=0
    #     )

    elif model == 'pixelsnail_bottom':
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
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=16)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--vqvae', type=str, default='ms-vqvae/vqvae_560.pt')  # 'small_ms-vqvae/vqvae_520.pt'
    # parser.add_argument('--top', type=str, default='cond-top/lr_0.0003/pixelsnail_top_300.pt')
    # parser.add_argument('--bottom', type=str, default='small_pixelsnail+/bottom_1500.pt')
    # parser.add_argument('--bottom', type=str, default='pixelsnail++/bottom_1093.pt')
    parser.add_argument('--bottom', type=str, default='pixelsnail-final/bottom_1401.pt')
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--label', type=int, default=3)
    parser.add_argument('--salience', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    # parser.add_argument('filename', type=str)

    args = parser.parse_args()

    if args.device is not None:
        device = 'cuda:' + args.device

    print(device)

    model_vqvae = load_model('vqvae', args.vqvae, device)
    # model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    for i in range(args.epoch):

        # top_sample = sample_model(model_top, device, args.batch, [10, 43], args.temp, condition=args.label)
        bottom_sample = sample_model(
            model_bottom, device, args.batch, [20, 86], args.temp, label_condition=args.label, salience_condition=args.salience)


        decoded_sample = model_vqvae.decode_code(bottom_sample)
        # print(decoded_sample.shape)

        out = decoded_sample.detach()
        out = out.cpu().numpy()

        for j, mel in enumerate(out):
            # file_name = os.path.join('sample-results/v2-result-2.0', str(args.label) + '-' + str(args.salience) + '-' + str((i + args.start_epoch) * args.batch + j))

            file_name = os.path.join(args.path, str(args.label) + '-' + str(args.salience) + '-' + str((i + args.start_epoch)*args.batch + j))

            print("save: ", file_name)
            np.save(file_name, mel)




