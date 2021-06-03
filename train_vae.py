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


def train(epoch, loader, model, optimizer, scheduler, device, args):
    loader = tqdm(loader)
    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 10

    mse_sum = 0
    mse_n = 0

    for i, (img, tgt) in enumerate(loader):
        model.zero_grad()
        img = img.to(device)
        tgt = tgt.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, tgt)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()

        if scheduler is not None and i % 200 == 0:
            scheduler.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}

        mse_sum += comm["mse_sum"]
        mse_n += comm["mse_n"]

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )

        if i % 100 == 0:
            model.eval()
            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            
            utils.save_image(
                torch.cat([tgt[:sample_size], out], 0),
                f"{args.outpath}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
                value_range=(-1, 1),
            )

            model.train()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.outpath):
        os.system(f'mkdir -p {args.outpath}')
    if not os.path.exists('./checkpoints'):
        os.system(f'mkdir -p ./checkpoints')


    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = DCFontStroke(src_path=args.srcpath, tgt_path=args.tgtpath, transform=transform, shift=True)
    loader = DataLoader(
        dataset, batch_size=32, num_workers=2, shuffle=True
    )

    if args.model == "VQVAE":
        model = VQVAE().to(device)

    if args.resume_path != "None":
        state_dict = torch.load(args.resume_path)
        model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3, eta_min=1e-5)

    for i in range(args.epoch):
        train(i, loader, model, optimizer, lr_scheduler, device, args)
        if (i + 1) % 5 == 0:
            torch.save(model.state_dict(), f"./checkpoints/{args.font}_size{args.size}_vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sched", type=str, default="cycle")
    parser.add_argument("--srcpath", type=str, default='./data/training_data/KAITI')
    parser.add_argument("--tgtpath", type=str, default='./data/training_data/SONG')
    parser.add_argument("--outpath", type=str, default='./example_results/styletrans_out_training')
    parser.add_argument("--model", type=str, default="VQVAE")
    parser.add_argument("--font", type=str, default='KaiSong')
    parser.add_argument("--resume_path", type=str, default="None")
    args = parser.parse_args()
    print(args)

    main(args)