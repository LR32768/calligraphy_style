from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections



def get_lim(img):
    arr = (np.array(img) > 0) * 255
    arr = arr[:, :, 0]
    h, w = arr.shape
    for i in range(h):
        if  not np.array_equal(arr[i, :], np.zeros(w)):
            lim_top = i
            break
    for i in range(h-1, 0, -1):
        if not np.array_equal(arr[i, :], np.zeros(w)):
            lim_bottom = i
            break
    for i in range(w):
        if not np.array_equal(arr[:, i], np.zeros(h)):
            lim_left = i
            break
    for i in range(w-1, 0, -1):
        if not np.array_equal(arr[:, i], np.zeros(h)):
            lim_right = i
            break
    return lim_left, lim_right, lim_top, lim_bottom


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    # img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    # draw = ImageDraw.Draw(img)
    # draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font, anchor='mm')
    img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (255, 255, 255), font=font, anchor='mm')
    return img


def draw_example(ch, dst_font, canvas_size, x_offset, y_offset):
    # dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    # example_img.paste(dst_img, (0, 0))
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    """
    arr = (np.array(dst_img) > 0) * 255
    arr = get_center(arr)
    dst_img = Image.fromarray(arr.astype('uint8')).convert('RGB')"""
    lim_left, lim_right, lim_top, lim_bottom = get_lim(dst_img)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    example_img.paste(dst_img, (0, 0))
    return example_img, lim_left, lim_right, lim_top, lim_bottom


data_idx = 19980
# plt.figure(figsize=(4, 4))
# ax = plt.gca()
for data_idx in range(19980, 20023 + 1):
    c = chr(data_idx)
    # dst_font = "./scatter_mat/kaiti_GB18030.ttf"
    # dst_font = "./scatter_mat/FZXKTJW.TTF"
    dst_font = "./scatter_mat/SIMKAI.TTF"
    t = 360
    dst_font = ImageFont.truetype(dst_font, size=t)
    canvas_size = t
    x_offset = canvas_size / 2
    y_offset = canvas_size / 2
    # x_offset = 0
    # y_offset = 0
    # sample_dir = './scatter_mat/figs/'

    e, lim_left, lim_right, lim_top, lim_bottom = draw_example(c, dst_font, canvas_size, x_offset, y_offset)


    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    image1 = e
    plt.imshow(image1)




    dataFile = f'./scatter_mat/data_mat/Font{data_idx}.mat'
    data = scio.loadmat(dataFile)

    x_min = np.inf
    x_max = 0
    y_min = np.inf
    y_max = 0
    for j in range(data['StrokeNum'][0, 0]):
        spn = data['StrokePointNum'][j, 0]
        x = data['StrokeSequenceX'][j, 1:spn].astype('int32')
        y = data['StrokeSequenceY'][j, 1:spn].astype('int32')
        if np.min(x) < x_min:
            x_min = np.min(x)
        if np.max(x) > x_max:
            x_max = np.max(x)
        if np.min(y) < y_min:
            y_min = np.min(y)
        if np.max(y) > y_max:
            y_max = np.max(y)
    print(x_min, x_max, y_min, y_max)

    alpha = 0.90
    for j in range(data['StrokeNum'][0, 0]):
        spn = data['StrokePointNum'][j, 0]
        x = data['StrokeSequenceX'][j, 1:spn].astype('int32')
        x = ((x - x_min) * ((lim_right - lim_left) /(x_max - x_min))) * alpha \
            + lim_left + (1 - alpha) * (lim_right - lim_left) / 2
        x = np.around(x)
        y = data['StrokeSequenceY'][j, 1:spn].astype('int32')
        y = ((y - y_min) * ((lim_bottom - lim_top) /(y_max - y_min))) * alpha \
            + lim_top + (1 - alpha) * (lim_bottom - lim_top) / 2
        y = np.around(y)
        plt.scatter(x, y, s=0.5)



    
    plt.xlim(0, t)
    plt.ylim(0, t)
    ax.set_xticks(np.arange(0, t, 50))
    ax.set_yticks(np.arange(0, t, 50))
    ax.set_xticklabels(np.arange(0, t, 50))
    ax.set_yticklabels(np.arange(0, t, 50))
    ax.invert_yaxis()
    plt.tick_params(top=True,bottom=False,left=True,right=False)
    plt.tick_params(labeltop=True,labelleft=True,labelright=False,labelbottom=False)
    plt.grid()
    plt.savefig(f'./scatter_mat/all/{data_idx}font.jpg')