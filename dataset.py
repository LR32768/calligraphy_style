import os
import csv
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def shift_image(img, sx, sy):
    """ Return the shifted image numpy array"""
    img_array = np.array(img)
    h, w, _ = img_array.shape
    canvas = np.zeros((img_array.shape[0] * 2, img_array.shape[1] * 2, 3))
    tx = int(sx * h + 0.5)
    ty = int(sy * w + 0.5)
    canvas[tx : tx+h, ty : ty+w, :] = img_array
    return canvas[h//2:h+h//2, w//2:w+w//2, :]

class DCFontStroke(Dataset):
    """
    A dataset class of [WholeCharacter, Stroke]
    """
    def __init__(self, src_path="./lesseq7/KAITI", tgt_path="./lesseq7/SONG", transform=None, train=True, shift=True):
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_data_list = []
        self.tgt_data_list = []
        self.train = train
        self.shift = shift

        if train: # Store source and target stroke list
            g = os.walk(src_path)
            for dpath, d_list, files in g:
                filtered_files = [f for f in files if len(f) > 16]
                for file in filtered_files:
                    if os.path.exists(os.path.join(tgt_path, file)):
                        self.src_data_list.append(os.path.join(dpath, file))
                        self.tgt_data_list.append(os.path.join(tgt_path, file))
        elif shift: # Store source and target whole font list
            g = os.walk(src_path)
            for dpath, d_list, files in g:
                filtered_files = [f for f in files if f.endswith('.csv')]
                for file in filtered_files:
                    if os.path.exists(os.path.join(tgt_path, file)):
                        self.src_data_list.append(os.path.join(dpath, file))
                        self.tgt_data_list.append(os.path.join(tgt_path, file))
        else:
            g = os.walk(src_path)
            for dpath, d_list, files in g:
                filtered_files = [f for f in files if len(f) < 22]
                for file in filtered_files:
                    if os.path.exists(os.path.join(tgt_path, file)):
                        self.src_data_list.append(os.path.join(dpath, file))
                        self.tgt_data_list.append(os.path.join(tgt_path, file))

        # print(self.src_data_list)
        # print(self.tgt_data_list)
        self.transform = transform

    def __len__(self):
        return len(self.src_data_list)

    def __getitem__(self, idx):

        if self.train: # Load each stroke image
            src_strk_path = self.src_data_list[idx]
            tgt_strk_path = self.tgt_data_list[idx]
            src_whole_path = src_strk_path[:-12]+'.jpg'
            src_csv_path = src_whole_path[:-4] + '.csv'
            tgt_csv_path = tgt_strk_path[:-12]+'.csv'
            with open(src_csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
                strk_idx = int(src_strk_path[-5])
                # breakpoint()
                tmp = rows[strk_idx]
                srcsy = float(tmp[0])
                srcsx = float(tmp[1])

            with open(tgt_csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
                strk_idx = int(src_strk_path[-5])
                tmp = rows[strk_idx]
                tgtsy = float(tmp[0])
                tgtsx = float(tmp[1])

            src_strk = Image.open(src_strk_path)
            tgt_strk = Image.open(tgt_strk_path)
            src_font = Image.open(src_whole_path)

            src_strk = shift_image(src_strk, srcsx, srcsy) / 255.0
            tgt_strk = shift_image(tgt_strk, tgtsx, tgtsy) / 255.0
            src_font = np.array(src_font) / 255.0

            if self.transform is not None:
                src_strk = self.transform(src_strk)
                tgt_strk = self.transform(tgt_strk)
                src_font = self.transform(src_font)
                return torch.cat((src_strk, src_font), 0).float(), tgt_strk.float()

        else: # Return a batch of [stroke_image, font_image] and target font image
            src_csv_path = self.src_data_list[idx]
            tgt_csv_path = self.tgt_data_list[idx]
            src_strk_imgs = []
            if self.shift:
                with open(src_csv_path, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = [row for row in reader]

                    for strk_idx in range(len(rows)):
                        tmp = rows[strk_idx]
                        srcsy = float(tmp[0])
                        srcsx = float(tmp[1])
                        src_strk_path = src_csv_path[:-4]+f'_stroke{strk_idx}.jpg'
                        img_ = Image.open(src_strk_path)
                        img_ = shift_image(img_, srcsx, srcsy) / 255.0
                        src_strk_imgs.append(img_)
            else:
                print(src_csv_path)
                num = int(src_csv_path[-6:-4])
                for strk_idx in range(num):
                    src_strk_path = src_csv_path[:-6] + f'stroke{strk_idx}.jpg'
                    img_ = Image.open(src_strk_path)
                    src_strk_imgs.append(img_)

            src_whole_path = src_csv_path[:-4]+'.jpg'
            tgt_whole_path = tgt_csv_path[:-4]+'.jpg'
            src_font = np.array(Image.open(src_whole_path)) / 255.0
            tgt_font = np.array(Image.open(tgt_whole_path)) / 255.0

            ret_batch = torch.zeros(len(src_strk_imgs), 6, 256, 256)
            for i in range(len(src_strk_imgs)):
                ret_batch[i] = torch.cat((self.transform(src_strk_imgs[i]), self.transform(src_font)), 0).float()
            return ret_batch, self.transform(src_font).float(), self.transform(tgt_font).float()


class DCFont(Dataset):
    """
    A dataset of [style1, style2]
    """
    def __init__(self, path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

if __name__ == "__main__":
    transform = transforms.ToTensor()
    dataset = DCFontStroke(transform=transform, train=False)
    im1, im2, im3 = dataset[109]
    print(len(dataset))
    #print(dataset[2])
    print(im1.size())
    print(im2.size())
    print(im3.size())
