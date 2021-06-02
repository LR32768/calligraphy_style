import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from dataset import DCFontStroke

from vqvae import VQVAE

def test(dataset, model, outpath, device):
    for i in range(len(dataset)):
        img, src, tgt = dataset[i]
        img = img.to(device)
        src = src.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)

        out, latent_loss = model(img)
        print(f'Rendering {i} th image ...')
        merge_out, _ = out.max(axis=0)
        #print(merge_out)
        #print(merge_out.size())
        utils.save_image(
                -torch.cat([src, tgt, merge_out.unsqueeze(0)], 0),
                os.path.join(outpath, f"font{i}.png"),
                nrow=3,
                normalize=True,
                range=(-1, 1),
        )


def main(args):
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = DCFontStroke(src_path = args.srcpath, tgt_path = args.tgtpath, transform=transform, train=False, shift = args.shift)

    model = VQVAE().to(device)
    state_dict = torch.load(args.weight_path)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        test(dataset, model, args.outpath, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sched", type=str, default="cycle")
    parser.add_argument("--srcpath", type=str, default='./lesseq7/KAITI')
    parser.add_argument("--tgtpath", type=str, default='./lesseq7/SONG')
    parser.add_argument("--outpath", type=str, default="./stroke_VAE/test_out")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--weight_path", type=str, default="./stroke_VAE/checkpoint/KaiSong_size256_vqvae_101.pt")
    args = parser.parse_args()
    print(args)

    main(args)