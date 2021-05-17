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
    """"""
    arr = (np.array(dst_img) > 0) * 255
    dst_img = Image.fromarray(arr.astype('uint8')).convert('RGB')
    example_img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    example_img.paste(dst_img, (0, 0))
    return example_img


data_idx = 19980
# plt.figure(figsize=(4, 4))
# ax = plt.gca()
for data_idx in range(19980, 19981):
    c = chr(data_idx)
    # dst_font = "./scatter_mat/kaiti_GB18030.ttf"
    dst_font = "./scatter_mat/FZXKTJW.TTF"
    t = 360
    dst_font = ImageFont.truetype(dst_font, size=t)
    canvas_size = t
    x_offset = canvas_size / 2
    y_offset = canvas_size / 2
    # x_offset = 0
    # y_offset = 0
    sample_dir = './scatter_mat/figs/'

    e = draw_example(c, dst_font, canvas_size, x_offset, y_offset)
    if e:
        e.save(os.path.join(sample_dir, "%d.jpg" % (ord(c))))


    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    image1 = Image.open(os.path.join(sample_dir, "%d.jpg" % (ord(c))))

    # width, height = image1.size   # Get dimensions
    # new_size = 280
    # left = (width - new_size)/2
    # top = (height - new_size)/2
    # right = (width + new_size)/2
    # bottom = (height + new_size)/2
    # # Crop the center of the image
    # image1 = image1.crop((left, top, right, bottom))
    plt.imshow(image1)

    
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