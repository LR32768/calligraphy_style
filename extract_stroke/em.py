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

import scipy
from scipy.stats import multivariate_normal
import matplotlib as mpl



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
    # total_cov = np.cov(data.transpose())
    # e, _ = np.linalg.eig(total_cov)
    # C = np.median(e) * np.eye(data.shape[1])
    C = np.eye(2) * 20
    sigma = np.concatenate([[C]] * n_components, axis=0)
    prior = np.concatenate([[1/n_components]] * n_components, axis=0)

    cont_flag = 1
    log_likelihood = 0
    # while cont_flag:
    for em_iter in range(3):
        print(f'em_iter:{em_iter}')

        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        canvas_size = 360
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

        plt.scatter(mu[:, 0], mu[:, 1], s=5)
        plt.scatter(data[:, 0], data[:, 1], s=0.5)
        for k in range(mu.shape[0]):
            make_ellipses(mean=mu[k], cov=sigma[k], ax=ax, alpha=0.1)
        plt.savefig(f'./extract_stroke/em_dynamic_fig/fig{em_iter}.jpg')

        sigma_new, prior_new = one_EM_iteration(data, n_components, mu, sigma, prior)
        sigma = sigma_new
        prior = prior_new
    return sigma, prior


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
        sigma_new[i] = 0.5 * sigma_new[i] + 0.5 * 20 *np.eye(2)
    
    prior_new = np.zeros(n_components)
    for i in range(n_components):
        prior_new[i] = np.sum(post_prob[:, i]) / N

    return sigma_new, prior_new


def post_prob_one_data(n_components, data, mu, sigma, prior):
    pdf = np.zeros(n_components)
    for i in range(n_components):
        pdf[i] = multivariate_normal.pdf(data, mean=mu[i], cov=sigma[i]) * prior[i]
    return pdf / np.sum(pdf)


def _get_pixel_coordinates(data_idx, canvas_size):
    c = chr(data_idx)
    font_dir = "./scatter_mat/SIMKAI.TTF"
    
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

    return pixel_x, pixel_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom


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
    n_stroke = data['StrokeNum'][0, 0]
    mu_class_idx = []
    for j in range(n_stroke):
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
            mu_class_idx.append(
                np.ndarray.tolist(
                    np.arange(0, n_components)
                )
            )
        else:
            n_components = n_components + x.shape[0]
            mu_newpart = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1)
            mu = np.concatenate((mu, mu_newpart), axis=0)
            mu_class_idx.append(
                np.ndarray.tolist(
                    np.arange(mu_class_idx[-1][-1] + 1, mu_class_idx[-1][-1] + 1 + x.shape[0])
                )
            )

    return mu, n_components, n_stroke, mu_class_idx


def make_ellipses(mean, cov, ax, confidence=5.991, alpha=0.3, color="blue", eigv=False, arrow_color_list=None):
    """
    多元正态分布
    mean: 均值
    cov: 协方差矩阵
    ax: 画布的Axes对象
    confidence: 置信椭圆置信率 # 置信区间， 95%： 5.991  99%： 9.21  90%： 4.605 
    alpha: 椭圆透明度
    eigv: 是否画特征向量
    arrow_color_list: 箭头颜色列表
    """
    lambda_, v = np.linalg.eig(cov)    # 计算特征值lambda_和特征向量v
    # print "lambda: ", lambda_
    # print "v: ", v
    # print "v[0, 0]: ", v[0, 0]

    sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值

    s = confidence
    width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
    height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
    angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
    ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color)    # 绘制椭圆

    ax.add_artist(ell)
    ell.set_alpha(alpha)


def get_stroke_points(pixel_full_x, pixel_full_y, n_components, n_stroke, mu, mu_class_idx, sigma, prior):
    data_all = np.concatenate(
        (np.expand_dims(pixel_full_x, 1), np.expand_dims(pixel_full_y, 1)),
        axis=1
        )

    data_each_stroke = []
    for _ in range(n_stroke):
        data_each_stroke.append([])

    for idx_data, data in enumerate(data_all):
        exponent = np.expand_dims(data - mu, 1)  # (n_components, 1, 2)
        exponent = np.einsum("ijk, ikl -> ijl" , exponent, np.linalg.inv(sigma))  # (n_components, 1, 2)
        exponent = np.einsum("ijk, ikl -> ijl" , exponent, np.expand_dims(data - mu, 2))  # (n_components, 1, 1)
        exponent = np.squeeze(exponent)  # (n_components,)
        exponent = (-0.5) * exponent
        # softmax_out = scipy.special.softmax(exponent)
        softmax_out = np.exp(exponent)
        pdf = np.divide(
            softmax_out,
            np.sqrt(4 * np.pi * np.pi * np.linalg.det(sigma))
        )
        breakpoint()
        pdf = np.multiply(pdf, prior)
        pdf = pdf / np.sum(pdf)
        breakpoint()

        
        topk_indices = pdf.argsort()[-2:][::-1]  # top 2 probability
        
        # breakpoint()
        for topk_index in topk_indices:
            # # TODO: for dynamic length len(mu_class_idx)
            # if topk_index in mu_class_idx[0]:
            #     data_each_stroke[0].append(np.ndarray.tolist(data))
            # elif topk_index in mu_class_idx[1]:
            #     data_each_stroke[1].append(np.ndarray.tolist(data))
            # elif topk_index in mu_class_idx[2]:
            #     data_each_stroke[2].append(np.ndarray.tolist(data))
            # elif topk_index in mu_class_idx[3]:
            #     data_each_stroke[3].append(np.ndarray.tolist(data))
            for idx_stroke, mu_class_sub_idx in enumerate(mu_class_idx):
                topk_index = topk_index - len(mu_class_sub_idx)
                if topk_index < 0:
                    data_each_stroke[idx_stroke].append(np.ndarray.tolist(data))
                    break
            """"""
    
    for idx_stroke in range(n_stroke):
        data_each_stroke[idx_stroke] = np.array(
            np.unique(
                data_each_stroke[idx_stroke],
                axis=0,
            )
        )

    return data_each_stroke






def main():
    data_idx_start = 19988
    for data_idx in range(data_idx_start, data_idx_start + 1):

        #####################################
        #  plt config                       #
        #####################################

        # plt.figure(figsize=(15, 15))
        # ax = plt.gca()
        canvas_size = 360
        # plt.xlim(0, canvas_size)
        # plt.ylim(0, canvas_size)
        # ax.set_xticks(np.arange(0, canvas_size, 50))
        # ax.set_yticks(np.arange(0, canvas_size, 50))
        # ax.set_xticklabels(np.arange(0, canvas_size, 50))
        # ax.set_yticklabels(np.arange(0, canvas_size, 50))
        # ax.invert_yaxis()
        # plt.tick_params(top=True, bottom=False, left=True, right=False)
        # plt.tick_params(labeltop=True, labelleft=True, labelright=False, labelbottom=False)
        # plt.grid()

        #####################################
        #  get pixel coordinates            #
        #####################################

        pixel_full_x, pixel_full_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom = _get_pixel_coordinates(data_idx, canvas_size)
        sparse_index = np.random.choice(np.arange(pixel_full_x.shape[0]), size=int(pixel_full_x.shape[0]/25),)
        sparse_index = np.sort(sparse_index)
        pixel_x = pixel_full_x[sparse_index]
        pixel_y = pixel_full_y[sparse_index]
        # pixel_x = get_sparce_points(pixel_x, step_size=20)
        # pixel_y = get_sparce_points(pixel_y, step_size=20)
        # plt.scatter(pixel_x, pixel_y, s=0.5)
        # plt.savefig(f'./extract_stroke/fig.jpg')

        

        #####################################
        #  downwards   get keypoints        #
        #           get mu, n_components    #
        #####################################
        mu, n_components, n_stroke, mu_class_idx = _get_mu_n(data_idx, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom)
        # plt.scatter(mu[:, 0], mu[:, 1], s=2)
        # for k in range(53):
        #     make_ellipses(mean=mu[k], cov=[[10, 0], [0, 10]], ax=ax, alpha=0.1)
        # plt.savefig(f'./extract_stroke/fig.jpg')
        # breakpoint()



        #####################################
        #  downwards           EM           #
        #####################################

        data_em = np.concatenate(
            (np.expand_dims(pixel_x, 1), np.expand_dims(pixel_y, 1)),
            axis=1
            )
        sigma, prior = em_mix(data=data_em, n_components=n_components, mu=mu, eps=None)

        stroke_points = get_stroke_points(pixel_full_x, pixel_full_y, n_components, n_stroke, mu, mu_class_idx, sigma, prior)

        # breakpoint()
        # plt.scatter(stroke_points[0][:, 0], stroke_points[0][:, 1], s=2, c='b')
        # plt.scatter(stroke_points[1][:, 0], stroke_points[1][:, 1], s=2, c='g')
        # plt.scatter(stroke_points[2][:, 0], stroke_points[2][:, 1], s=2, c='r')
        # plt.scatter(stroke_points[3][:, 0], stroke_points[3][:, 1], s=2, c='m')
        plt.scatter(stroke_points[4][:, 0], stroke_points[4][:, 1], s=2, c='c')
        plt.savefig(f'./extract_stroke/em_dynamic_fig/siged_full55.jpg')


if __name__ == "__main__":
    main()