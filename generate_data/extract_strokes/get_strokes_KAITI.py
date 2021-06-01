from __future__ import print_function
from __future__ import absolute_import

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

import argparse

# some tools
def plot_ellipses(mean, cov, ax, confidence=5.991, alpha=0.3, color="blue"):

    lambda_, v = np.linalg.eig(cov)    # 计算特征值和特征向量v
    sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值

    s = confidence
    width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
    height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
    angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
    ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color)    # 绘制椭圆

    ax.add_artist(ell)
    ell.set_alpha(alpha)


def get_sparce_points(x, step_size=10):
    length = x.shape[0]
    select_index = np.arange(0, length, step_size)
    return x[select_index]


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


def get_pixel_coordinates(data_idx, canvas_size):

    c = chr(data_idx)
    """ 楷体 """
    font_dir = "./data/fonts/KAITI.ttf"  # good

    """ 宋体 """
    # font_dir = "./data/fonts/SIMFANG.TTF"
    
    font = ImageFont.truetype(font_dir, size=canvas_size)

    font_array= _get_font_array(
        ch=c,
        font=font,
        canvas_size=canvas_size,
        x_offset=canvas_size/2,
        y_offset=canvas_size/2
        )
    font_lim_left, font_lim_right, font_lim_top, font_lim_bottom = _get_font_array_lim(font_array)

    pixel_x, pixel_y = _font_img_2_xy(font_array)

    return pixel_x, pixel_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom


# get mu coordinates & related tools
def get_mu_n(data_idx, mu_sparsity, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom):
    # dataFile = f'/cluster/home/pyf/code/calligraphy_style/data_kai_GB18030/Font{data_idx}.mat'
    dataFile = f'./data/KAITI_skeleton/Font{data_idx}.mat'
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

    alpha_x = 0.9
    alpha_y = 0.9
    n_stroke = data['StrokeNum'][0, 0]
    mu_class_idx = []
    for j in range(n_stroke):
        spn = data['StrokePointNum'][j, 0]
        x = data['StrokeSequenceX'][j, 1:spn].astype('int32')
        x = ((x - x_min) * ((font_lim_right - font_lim_left) /(x_max - x_min))) * alpha_x \
            + font_lim_left + (1 - alpha_x) * (font_lim_right - font_lim_left) / 2
        x = np.around(x)
        y = data['StrokeSequenceY'][j, 1:spn].astype('int32')
        y = ((y - y_min) * ((font_lim_bottom - font_lim_top) /(y_max - y_min))) * alpha_y \
            + font_lim_top + (1 - alpha_y) * (font_lim_bottom - font_lim_top) / 2
        y = np.around(y)

        x = get_sparce_points(x, step_size=mu_sparsity)
        y = get_sparce_points(y, step_size=mu_sparsity)
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


# Expectation–maximization algorithm & related tools
def affine_transform(mu_new, mu, n_components, mu_init, idx_stroke):
    mp_inverse = np.linalg.pinv(
        np.concatenate(
            (
                np.expand_dims(mu[:,0], axis=1),
                np.expand_dims(mu[:,1], axis=1),
                np.expand_dims(np.ones(n_components), axis=1),
            ),
            axis=1
        )
    )
    ab1 = np.matmul(
        mp_inverse,
        np.expand_dims(mu_new[:,0], axis=1)
    )
    ab2 = np.matmul(
        mp_inverse,
        np.expand_dims(mu_new[:,1], axis=1)
    )
    affine = np.concatenate(
        (ab1[:2].transpose(), ab2[:2].transpose()),
        axis=0
    )
    offset = np.concatenate(
        (ab1[-1].transpose(), ab2[-1].transpose()),
        axis=0
    )
    mu_tmp = np.matmul(
        mu,
        affine.transpose()
    ) + offset
    mu_new = mu_tmp
    return mu_new


def one_EM_iteration(data, n_components, mu, sigma, prior, n_stroke, mu_class_idx, mu_init):
    N, d = data.shape

    # E-step
    post_prob = np.zeros((N, n_components))
    for idx_mu in range(n_components):
        post_prob[:, idx_mu] = multivariate_normal.pdf(x=data, mean=mu[idx_mu], cov=sigma[idx_mu]) * prior[idx_mu]
    post_prob = post_prob / np.sum(post_prob, axis=1)[:, np.newaxis]

    # M-step
    mu_new = np.zeros((n_components, d))
    for i in range(n_components):
        mu_new[i] = np.matmul(
            np.expand_dims(post_prob[:,i], 0), data
        )
        mu_new[i] = np.squeeze(mu_new[i])
        mu_new[i] = mu_new[i] / np.sum(post_prob[:,i])
    # mu_new = affine_transform(mu_new, mu, n_components)
    for idx_stroke in range(n_stroke):
        mu_new[mu_class_idx[idx_stroke]] = affine_transform(mu_new[mu_class_idx[idx_stroke]],
                                                            mu[mu_class_idx[idx_stroke]],
                                                            n_components=len(mu_class_idx[idx_stroke]),
                                                            mu_init=mu_init[mu_class_idx[idx_stroke]],
                                                            idx_stroke=idx_stroke)

    sigma_new = np.zeros((n_components, d, d))
    for i in range(n_components):
        data_centered = data - mu[i]
        sigma_tmp = np.einsum("ijk, ikl -> ijl", np.expand_dims(data_centered, 2), np.expand_dims(data_centered, 1))
        sigma_tmp = post_prob[:,i] * np.reshape(sigma_tmp, (N, d*d)).transpose()
        sigma_tmp = np.reshape(sigma_tmp.transpose(), (N, d, d))
        sigma_new[i] = np.sum(sigma_tmp, axis=0) / np.sum(post_prob[:,i])
        sigma_new[i] = 0.5 * sigma_new[i] + 0.5 * 20 *np.eye(2)
    sigma_smooth = np.zeros((n_components, d, d))
    for i in range(n_components):
        if i == 0:
            sigma_smooth[i] = 0.7 * sigma_new[i] + 0.3 * sigma_new[i+1]
        elif i == n_components - 1:
            sigma_smooth[i] = 0.7 * sigma_new[i] + 0.3 * sigma_new[i-1]
        else:
            sigma_smooth[i] = 0.3 * sigma_new[i-1] + 0.4 * sigma_new[i] + 0.3 * sigma_new[i+1]
    sigma_new = sigma_smooth
    
    prior_new = np.sum(post_prob, axis=0) / N

    return mu_new, sigma_new, prior_new


def em_mix(data, n_components, mu_class_idx, n_stroke, mu, canvas_size):
    #  runs EM to estimate a Gaussian mixture model.
    #
    #  data: input data, N * dims(2) numpy array.
    #  n_components: number_of_components, equals to the number of keypoints, int.
    #  mu: the mean of each components, is the coordinate of each keypoints,
    #       numpy array, n_components * 2


    #  initial params
    C = np.eye(2) * 20
    sigma = np.concatenate([[C]] * n_components, axis=0)
    prior = np.concatenate([[1/n_components]] * n_components, axis=0)
    mu_init = np.copy(mu)

    _mu = np.copy(mu)

    # while cont_flag:
    for em_iter in range(20):
        start = time.time()
        print(f'em_iter:{em_iter}')

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
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
            plot_ellipses(mean=mu[k], cov=sigma[k], ax=ax, alpha=0.1)
        plt.savefig(f'./example_results/strokes/process_of_extract_KAITI/process_EM/em_iter-{em_iter}.jpg')
        plt.close()


        mu_new, sigma_new, prior_new = one_EM_iteration(data, n_components, _mu, sigma, prior, n_stroke, mu_class_idx, mu_init)
        mu = mu_new
        sigma = sigma_new
        prior = prior_new
        print(f'time{em_iter}:{time.time() - start}')

    return mu, sigma, prior


# font stroke segmentation
def get_stroke_points(pixel_full_x, pixel_full_y, n_components, n_stroke, mu, mu_class_idx, sigma, prior, threshold):
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
        pdf = np.divide(
            np.exp(exponent),
            np.sqrt(4 * np.pi * np.pi * np.linalg.det(sigma))
        )
        pdf = np.multiply(pdf, prior)
        pdf = pdf / np.sum(pdf)

        for idx_stroke in range(n_stroke):
            prob_each_stroke = np.sum(
                pdf[mu_class_idx[idx_stroke]]
            )
            if prob_each_stroke > threshold:
                data_each_stroke[idx_stroke].append(np.ndarray.tolist(data))     
    
    for idx_stroke in range(n_stroke):
        data_each_stroke[idx_stroke] = np.array(
            np.unique(
                data_each_stroke[idx_stroke],
                axis=0,
            )
        )

    return data_each_stroke


# plot stroke images
def plot_stroke(data_idx, stroke_points, n_stroke, canvas_size):
    for idx_stroke in range(n_stroke):
        # get centroi
        centroi_x = np.sum(stroke_points[idx_stroke][:, 0]) / stroke_points[idx_stroke][:, 0].shape[0]
        centroi_y = np.sum(stroke_points[idx_stroke][:, 1]) / stroke_points[idx_stroke][:, 1].shape[0]
        x = stroke_points[idx_stroke][:, 0] - centroi_x + (canvas_size / 2)
        y = stroke_points[idx_stroke][:, 1] - centroi_y + (canvas_size / 2)

        if idx_stroke == 0:
            with open(f'./example_results/strokes/strokes_KAITI/KAITI_unicode{data_idx}.csv', 'w') as f:
                f.write(f"{centroi_x/canvas_size:.4f},{centroi_y/canvas_size:.4f},\n")
        else:
            with open(f'./example_results/strokes/strokes_KAITI/KAITI_unicode{data_idx}.csv', 'a') as f:
                f.write(f"{centroi_x/canvas_size:.4f},{centroi_y/canvas_size:.4f},\n")

        plt.figure(figsize=(3.05, 3.075), facecolor='black')
        ax = plt.gca()
        plt.xlim(0, canvas_size)
        plt.ylim(0, canvas_size)
        ax.invert_yaxis()
        plt.axis('off')
        plt.scatter(x, y, s=2, c='white')
        

        plt.savefig(f'./example_results/strokes/strokes_KAITI/KAITI_unicode{data_idx}_stroke{idx_stroke}.jpg', bbox_inches='tight', dpi=100)


def main(args):
    data_idx_start = args.unicode
    data_idx_end = data_idx_start
    for data_idx in range(data_idx_start, data_idx_end + 1):

        # config
        canvas_size = 360
        em_data_sparsity = 10
        mu_sparsity = 5
        threshold = 0.2

        # get pixel coordinates
        pixel_full_x, pixel_full_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom = get_pixel_coordinates(data_idx, canvas_size)
        sparse_index = np.random.choice(
            np.arange(pixel_full_x.shape[0]),
            size=int(pixel_full_x.shape[0]/em_data_sparsity),
        )
        sparse_index = np.sort(sparse_index)
        pixel_x = pixel_full_x[sparse_index]
        pixel_y = pixel_full_y[sparse_index]
   

        # get mu, 
        #     number of Gaussian Mixture Model compoments(num of mu),
        #     number of stroke, 
        #     mu each stroke's indices
        mu, n_components, n_stroke, mu_class_idx = get_mu_n(data_idx, mu_sparsity, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom)

        # EM
        data_em = np.concatenate(
            (np.expand_dims(pixel_x, 1), np.expand_dims(pixel_y, 1)),
            axis=1,
            )
        mu, sigma, prior = em_mix(data=data_em, n_components=n_components, mu=mu, mu_class_idx=mu_class_idx,
                                n_stroke=n_stroke, canvas_size=canvas_size)


        # stroke segmentation
        stroke_points = get_stroke_points(pixel_full_x, pixel_full_y, n_components, n_stroke, mu, mu_class_idx, sigma, prior, threshold)

        # plot stroke images
        plot_stroke(data_idx, stroke_points, n_stroke, canvas_size)
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unicode', type=int)
    args = parser.parse_args()

    main(args)