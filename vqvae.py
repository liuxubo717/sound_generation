import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks_1 = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

            blocks_2 = [
                nn.Conv2d(in_channel, channel // 2, 2, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 2, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

            blocks_3 = [
                nn.Conv2d(in_channel, channel // 2, 6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 6, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

            blocks_4 = [
                nn.Conv2d(in_channel, channel // 2, 8, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 8, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        # elif stride == 2:
        #     blocks_1 = [
        #         nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(channel // 2, channel, 3, padding=1),
        #     ]
        #
        #     blocks_2 = [
        #         nn.Conv2d(in_channel, channel // 2, 2, stride=2, padding=0),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(channel // 2, channel, 3, padding=1),
        #     ]
        #
        #     blocks_3 = [
        #         nn.Conv2d(in_channel, channel // 2, 8, stride=2, padding=3),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(channel // 2, channel, 3, padding=1),
        #     ]
        #
        #     blocks_4 = [
        #         nn.Conv2d(in_channel, channel // 2, 16, stride=2, padding=7),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(channel // 2, channel, 3, padding=1),
        #     ]

        for i in range(n_res_block):
            blocks_1.append(ResBlock(channel, n_res_channel))
            blocks_2.append(ResBlock(channel, n_res_channel))
            blocks_3.append(ResBlock(channel, n_res_channel))
            blocks_4.append(ResBlock(channel, n_res_channel))

        blocks_1.append(nn.ReLU(inplace=True))
        blocks_2.append(nn.ReLU(inplace=True))
        blocks_3.append(nn.ReLU(inplace=True))
        blocks_4.append(nn.ReLU(inplace=True))

        self.blocks_1 = nn.Sequential(*blocks_1)
        self.blocks_2 = nn.Sequential(*blocks_2)
        self.blocks_3 = nn.Sequential(*blocks_3)
        self.blocks_4 = nn.Sequential(*blocks_4)

    def forward(self, input):
        return self.blocks_1(input) + self.blocks_2(input) + self.blocks_3(input) + self.blocks_4(input)

        # return self.blocks_1(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channel=1,  # for mel-spec.
            channel=128,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_embed=512,
            decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        # self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        # self.quantize_t = Quantize(embed_dim, n_embed)
        # self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        # self.upsample_t = nn.ConvTranspose2d(
        #     embed_dim, embed_dim, 4, stride=2, padding=1
        # )
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_b, diff, _ = self.encode(input)
        dec = self.decode(quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        # enc_t = self.enc_t(enc_b)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_b, diff_b, id_b

    def decode(self, quant_b):
        # _dec = self.dec_t(quant_t)
        dec = self.dec(quant_b)

        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_b)

        return dec


if __name__ == '__main__':
    import audio2mel
    from datasets import get_dataset_filelist
    from torch.utils.data import DataLoader

    train_file_list, _ = get_dataset_filelist()

    train_set = audio2mel.Audio2Mel(train_file_list[0:4], 22050 * 4, 1024, 80, 256, 22050, 0, 8000)

    loader = DataLoader(train_set, batch_size=2, sampler=None, num_workers=2)

    model = VQVAE()

    a = torch.randn(3, 3).to('cuda')
    print(a)
    model = model.to('cuda')

    for i, batch in enumerate(loader):
        mel, id, name = batch
        mel = mel.to('cuda')
        out, latent_loss = model(mel)
        print(out.shape)
        if i == 5:
            break
