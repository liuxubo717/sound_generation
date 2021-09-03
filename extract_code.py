import argparse
import pickle

import torch
from torch.utils.data import DataLoader
import lmdb
from tqdm import tqdm
import audio2mel
from datasets import get_dataset_filelist
from vqvae import VQVAE
from collections import namedtuple
from datasets import CodeRow

# CodeRow = namedtuple('CodeRow', ['bottom', 'class_id', 'filename'])

def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, class_id, salience, filename in pbar:
            img = img.to(device)

            _, _, id_b = model.encode(img)
            # id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for c_id, sali, file, bottom in zip(class_id, salience, filename, id_b):
                row = CodeRow(bottom=bottom, class_id=c_id, salience=sali, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoint/ms-vqvae/vqvae_560.pt')
    parser.add_argument('--name', type=str, default='vqvae-code')

    args = parser.parse_args()

    device = 'cuda'

    train_file_list, _ = get_dataset_filelist()

    train_set = audio2mel.Audio2Mel(train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000)

    loader = DataLoader(train_set, batch_size=128, sampler=None, num_workers=2)

    # for i, batch in enumerate(loader):l
    #     mel, id, name = batch

    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)
