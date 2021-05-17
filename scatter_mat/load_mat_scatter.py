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
    img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (255, 255, 255), font=font, anchor='mm')
    return img


def draw_example(ch, dst_font, canvas_size, x_offset, y_offset):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    example_img.paste(dst_img, (0, 0))
    return example_img


data_idx = 19980 + 2
c = chr(data_idx)
dst_font = "./scatter_mat/FZXKTJW.TTF"
t = 360
dst_font = ImageFont.truetype(dst_font, size=t)
canvas_size = t
x_offset = canvas_size / 2
y_offset = canvas_size / 2
sample_dir = './scatter_mat/'

e = draw_example(c, dst_font, canvas_size, x_offset, y_offset)
if e:
    e.save(os.path.join(sample_dir, "fig.jpg"))


plt.figure()
image1 = Image.open(os.path.join(sample_dir, "fig.jpg"))

# width, height = image1.size   # Get dimensions
# new_size = 280
# left = (width - new_size)/2
# top = (height - new_size)/2
# right = (width + new_size)/2
# bottom = (height + new_size)/2
# # Crop the center of the image
# image1 = image1.crop((left, top, right, bottom))

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


for j in range(data['StrokeNum'][0, 0]):
    spn = data['StrokePointNum'][j, 0]
    x = data['StrokeSequenceX'][j, 1:spn].astype('int32')
    x = (x - x_min) * (300 /(x_max - x_min))
    x = np.around(x) + 30
    y = data['StrokeSequenceY'][j, 1:spn].astype('int32')
    y = (y - y_min) * (300 /(y_max - y_min))
    y = np.around(y) + 30
    plt.scatter(x, y, s=0.5)

ax = plt.gca()
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
plt.savefig('./scatter_mat/pic.jpg')