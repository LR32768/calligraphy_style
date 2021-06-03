from __future__ import print_function
from __future__ import absolute_import

import os
import time
import math
import numpy as np

import scipy
import scipy.io as scio
from scipy.stats import multivariate_normal

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import cv2
from skimage import morphology

import argparse


# get_pixel_coordinates & related tools
def _draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (255, 255, 255), font=font, anchor='mm')
    return img


def _get_font_array(ch, font, canvas_size, x_offset, y_offset):
    font_img = _draw_single_char(ch, font, canvas_size, x_offset, y_offset)
    return np.array(font_img)


def _get_font_array_lim(font_array):
    arr = (font_array > 0) * 255
    arr = arr[:, :, 0]
    h, w = arr.shape
    for i in range(h):
        if  not np.array_equal(arr[i, :], np.zeros(w)):
            font_lim_top = i
            break
    for i in range(h-1, 0, -1):
        if not np.array_equal(arr[i, :], np.zeros(w)):
            font_lim_bottom = i
            break
    for i in range(w):
        if not np.array_equal(arr[:, i], np.zeros(h)):
            font_lim_left = i
            break
    for i in range(w-1, 0, -1):
        if not np.array_equal(arr[:, i], np.zeros(h)):
            font_lim_right = i
            break
    return font_lim_left, font_lim_right, font_lim_top, font_lim_bottom


def _font_img_2_xy(font_array):
    pixel_x = np.nonzero(font_array[:,:,0])[1]
    pixel_y = np.nonzero(font_array[:,:,0])[0]  # 1, 0 wired!
    return pixel_x, pixel_y


def _font_img_2_xy_1d(font_array):
    pixel_x = np.nonzero(font_array)[1]
    pixel_y = np.nonzero(font_array)[0]  # 1, 0 wired!
    return pixel_x, pixel_y


def get_pixel_coordinates(data_idx, canvas_size, font, skeletonize=False):

    c = chr(data_idx)

    if font == 'KAITI':  # 楷体
        font_dir = "./data/fonts/KAITI.ttf"  # good
    elif font == 'SONG':  # 宋体
        font_dir = "./data/fonts/SIMFANG.TTF"
    else:
        raise NotImplemented
    
    font = ImageFont.truetype(font_dir, size=canvas_size)

    font_array= _get_font_array(
        ch=c,
        font=font,
        canvas_size=canvas_size,
        x_offset=canvas_size/2,
        y_offset=canvas_size/2
        )
    font_lim_left, font_lim_right, font_lim_top, font_lim_bottom = _get_font_array_lim(font_array)

    if skeletonize == False:
        pixel_x, pixel_y = _font_img_2_xy(font_array)
    else:
        _, binary = cv2.threshold(font_array, 200, 255, cv2.THRESH_BINARY)
        binary[binary==255] = 1
        skeleton0 = morphology.skeletonize(binary)
        skeleton = skeleton0.astype(np.uint8) * 255
        skeleton = np.max(skeleton, axis=2)
        pixel_x, pixel_y = _font_img_2_xy_1d(skeleton)

    return pixel_x, pixel_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom


def plot_stroke(x, y, canvas_size, data_idx, font):
    plt.figure(figsize=(3.05, 3.075), facecolor='black')
    ax = plt.gca()
    plt.xlim(0, canvas_size)
    plt.ylim(0, canvas_size)
    ax.invert_yaxis()
    plt.axis('off')
    plt.scatter(x, y, s=2, c='white')

    plt.savefig(f'./example_results/whole_fonts/{font}/unicode{data_idx}_{font}.jpg', bbox_inches='tight', dpi=100)
    plt.close()


def main(args):
    data_idx = args.unicode
    canvas_size = 360
    font_style = args.font_style
    pixel_full_x, pixel_full_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom = \
        get_pixel_coordinates(data_idx, canvas_size, font=font_style, skeletonize=False)
    plot_stroke(pixel_full_x, pixel_full_y, canvas_size, data_idx, font=font_style)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unicode', type=int)
    parser.add_argument('--font_style', type=str)
    args = parser.parse_args()

    if not os.path.exists('./example_results/whole_fonts/KAITI/'):
        os.system('mkdir -p ./example_results/whole_fonts/KAITI/')
    if not os.path.exists('./example_results/whole_fonts/SONG/'):
        os.system('mkdir -p ./example_results/whole_fonts/SONG/')

    main(args)