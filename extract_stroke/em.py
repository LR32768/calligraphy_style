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

from scipy.stats import multivariate_normal



def get_font_array_lim(font_array):
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


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (255, 255, 255), font=font, anchor='mm')
    return img


def get_font_array(ch, font, canvas_size, x_offset, y_offset):
    font_img = draw_single_char(ch, font, canvas_size, x_offset, y_offset)
    return np.array(font_img)


def font_img_2_xy(font_array):
    pixel_x = np.nonzero(font_array[:,:,0])[1]
    pixel_y = np.nonzero(font_array[:,:,0])[0]  # 1, 0 wired!
    return pixel_x, pixel_y


def get_sparce_points(x, step_size=10):
    length = x.shape[0]
    select_index = np.arange(0, length, step_size)
    return x[select_index]





def em_mix(data, n_components, mu, eps=None):
    #  runs EM to estimate a Gaussian mixture model.
    #
    #
    #  data: input data, N * dims(2) numpy array.
    #  n_components: number_of_components, equals to the number of keypoints, int.
    #  mu: the mean of each components, is the coordinate of each keypoints,
    #       numpy array, n_components * 2
    #  eps(optional): stopping criterion, float.
    #
    #  param: params of different gaussians, list.
    #  history(optional): params of different gaussians during iteration, dict.
    #  ll(optional): log-likelihood of the data during iteration, list.

    # set stopping  criterion
    if eps == None:
        eps = min(1e-3, 1/(data.shape[0]*100))

    #  initial params
    total_cov = np.cov(data.transpose())
    e, _ = np.linalg.eig(total_cov)
    C = np.median(e) * np.eye(data.shape[1])
    sigma = np.concatenate([[C]] * n_components, axis=0)
    prior = np.concatenate([[1/n_components]] * n_components, axis=0)

    cont_flag = 1
    log_likelihood = 0
    # while cont_flag:
    for _ in range(10):
        sigma_new = one_EM_iteration(data, n_components, mu, sigma, prior)
        print(np.linalg.norm(sigma_new - sigma_new))
        sigma = sigma_new


def one_EM_iteration(data, n_components, mu, sigma, prior):
    N, d = data.shape
    post_prob = np.zeros((N, n_components))
    for i in range(N):
        # print(f'i in E step{i}')
        post_prob[i] = post_prob_one_data(n_components, data[i], mu, sigma, prior)

    # M-step
    sigma_new = np.zeros((n_components, d, d))
    for i in range(n_components):
        data_centered = data - mu[i]
        sigma_tmp = np.einsum("ijk, ikl -> ijl", np.expand_dims(data_centered, 2), np.expand_dims(data_centered, 1))
        sigma_tmp = post_prob[:,i] * np.reshape(sigma_tmp, (N, d*d)).transpose()
        sigma_tmp = np.reshape(sigma_tmp.transpose(), (N, d, d))
        sigma_new[i] = np.sum(sigma_tmp, axis=0)
        sigma_new[i] = sigma_new[i] / np.sum(post_prob[:,i])
    return sigma_new


def post_prob_one_data(n_components, data, mu, sigma, prior):
    pdf = np.zeros(n_components)
    for i in range(n_components):
        pdf[i] = multivariate_normal.pdf(data, mean=mu[i], cov=sigma[i]) * prior[i]
    return pdf / np.sum(pdf)


def _get_pixel_coordinates(data_idx):
    c = chr(data_idx)
    font_dir = "./scatter_mat/SIMKAI.TTF"
    canvas_size = 360
    font = ImageFont.truetype(font_dir, size=canvas_size)

    font_array= get_font_array(
        ch=c,
        font=font,
        canvas_size=canvas_size,
        x_offset=canvas_size/2,
        y_offset=canvas_size/2
        )
    font_lim_left, font_lim_right, font_lim_top, font_lim_bottom = get_font_array_lim(font_array)

    pixel_x, pixel_y = font_img_2_xy(font_array)

    return pixel_x, pixel_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom, canvas_size


def _get_mu_n(data_idx, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom):
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
    # print(x_min, x_max, y_min, y_max)

    alpha = 0.90
    # n_components = 0
    for j in range(data['StrokeNum'][0, 0]):
        spn = data['StrokePointNum'][j, 0]
        x = data['StrokeSequenceX'][j, 1:spn].astype('int32')
        x = ((x - x_min) * ((font_lim_right - font_lim_left) /(x_max - x_min))) * alpha \
            + font_lim_left + (1 - alpha) * (font_lim_right - font_lim_left) / 2
        x = np.around(x)
        y = data['StrokeSequenceY'][j, 1:spn].astype('int32')
        y = ((y - y_min) * ((font_lim_bottom - font_lim_top) /(y_max - y_min))) * alpha \
            + font_lim_top + (1 - alpha) * (font_lim_bottom - font_lim_top) / 2
        y = np.around(y)

        x = get_sparce_points(x)
        y = get_sparce_points(y)
        if j == 0:
            n_components = x.shape[0]
            mu = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1)
        else:
            n_components = n_components + x.shape[0]
            mu_newpart = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1)
            mu = np.concatenate((mu, mu_newpart), axis=0)

    return mu, n_components



def main():
    data_idx = 19980
    for data_idx in range(19980, 19980 + 1):
        plt.figure(figsize=(4, 4))
        ax = plt.gca()
        #####################################
        #  downwards  get pixel coordinates #
        #####################################

        pixel_x, pixel_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom, canvas_size = _get_pixel_coordinates(data_idx)

        plt.scatter(pixel_x, pixel_y)
        plt.savefig(f'./extract_stroke/fig.jpg')
        breakpoint()
        

        #####################################
        #  downwards   get keypoints        #
        #           get mu, n_components    #
        #####################################
        mu, n_components = _get_mu_n(data_idx, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom)


        #####################################
        #  downwards     plot configs       #
        #####################################
        plt.xlim(0, canvas_size)
        plt.ylim(0, canvas_size)
        ax.set_xticks(np.arange(0, canvas_size, 50))
        ax.set_yticks(np.arange(0, canvas_size, 50))
        ax.set_xticklabels(np.arange(0, canvas_size, 50))
        ax.set_yticklabels(np.arange(0, canvas_size, 50))
        ax.invert_yaxis()
        plt.tick_params(top=True, bottom=False, left=True, right=False)
        plt.tick_params(labeltop=True, labelleft=True, labelright=False, labelbottom=False)
        plt.grid()
        # plt.savefig(f'./extract_stroke/{data_idx}font.jpg')


        #####################################
        #  downwards           EM           #
        #####################################
        pixel_x = get_sparce_points(pixel_x, step_size=100)
        pixel_y = get_sparce_points(pixel_y, step_size=100)
        plt.scatter(pixel_x, pixel_y)
        plt.savefig(f'./extract_stroke/fig.jpg')
        data_em = np.concatenate(
            (np.expand_dims(pixel_x, 1), np.expand_dims(pixel_y, 1)),
            axis=1
            )
        em_mix(data=data_em, n_components=n_components, mu=mu, eps=None)


if __name__ == "__main__":
    main()