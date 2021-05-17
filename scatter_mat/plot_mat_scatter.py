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



# data_idx = 19980
# plt.figure(figsize=(4, 4))
# ax = plt.gca()
for data_idx in range(19980, 19980 + 1):
    dataFile = f'./scatter_mat/data_mat/Font{data_idx}.mat'
    data = scio.loadmat(dataFile)

    print(np.max(data['StrokeSequenceX']))
    print(np.min(data['StrokeSequenceX']))
    print(np.max(data['StrokeSequenceY']))
    print(np.min(data['StrokeSequenceY']))

    # plt.figure(figsize=(4, 4))
    # ax = plt.gca()

    # for j in range(data['StrokeNum'][0, 0]):
    #     spn = data['StrokePointNum'][j, 0]
    #     x = data['StrokeSequenceX'][j, 1:spn]
    #     y = data['StrokeSequenceY'][j, 1:spn]
    #     plt.scatter(x, y, s=0.5)

    # plt.xlim(0, 350)
    # plt.ylim(0, 350)
    # ax.set_xticks(np.arange(0, 350, 50))
    # ax.set_yticks(np.arange(0, 350, 50))
    # ax.set_xticklabels(np.arange(0, 350, 50))
    # ax.set_yticklabels(np.arange(0, 350, 50))
    # ax.invert_yaxis()
    # plt.tick_params(top=True,bottom=False,left=True,right=False)
    # plt.tick_params(labeltop=True,labelleft=True,labelright=False,labelbottom=False)
    # plt.grid()
    # plt.savefig(f'./scatter_mat/all/{data_idx}scatter.jpg')