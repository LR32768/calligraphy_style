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




# some tools
def plot_ellipses(mean, cov, ax, confidence=5.991, alpha=0.3, color_idx=1, color="blue"):
    color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 
                  'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                  'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',]


    lambda_, v = np.linalg.eig(cov)    # 计算特征值lambda_和特征向量v

    sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值

    s = confidence
    width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
    height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
    angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
    ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color_list[color_idx])    # 绘制椭圆

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


def _font_img_2_xy_1d(font_array):
    pixel_x = np.nonzero(font_array)[1]
    pixel_y = np.nonzero(font_array)[0]  # 1, 0 wired!
    return pixel_x, pixel_y


def get_pixel_coordinates(data_idx, canvas_size, font, skeletonize=False):

    c = chr(data_idx)

    if font == 'KAI':  # 楷体
        font_dir = "./data/fonts/KAITI.ttf"  # good
    elif font == 'SONG':  # 宋体
        font_dir = "./data/fonts/SIMFANG.TTF"
    
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
    A = np.concatenate(
        (
            np.expand_dims(mu_init[:,0], axis=1),
            np.expand_dims(mu_init[:,1], axis=1),
            np.expand_dims(np.ones(n_components), axis=1),
        ),
        axis=1
    )
    lbd = 100
    A2 = np.linalg.inv(2 * np.matmul(A.transpose(), A) + lbd * np.eye(3))
    X = np.zeros((2, 3))
    b1 = 2 * A.transpose() @ mu_new[:, 0]; b1[0] += lbd
    X[0] = A2 @ b1
    b2 = 2 * A.transpose() @ mu_new[:, 1]; b2[1] += lbd
    X[1] = A2 @ b2

    return A @ X.transpose()


def one_EM_iteration(data, n_components, mu, sigma, prior, n_stroke, mu_class_idx, mode, mu_init):
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
    for idx_stroke in range(n_stroke):
        mu_new[mu_class_idx[idx_stroke]] = affine_transform(mu_new[mu_class_idx[idx_stroke]],
                                                            mu[mu_class_idx[idx_stroke]],
                                                            n_components=len(mu_class_idx[idx_stroke]),
                                                            mu_init=mu_init[mu_class_idx[idx_stroke]],
                                                            idx_stroke=idx_stroke)
    # mu_new = affine_transform(mu_new, mu, n_components)
    # mu_smooth = np.zeros((n_components, d))
    # for i in range(n_components):
    #     if i == 0:
    #         mu_smooth[i] = 0.99 * mu_new[i] + 0.01 * mu_new[i+1]
    #     elif i == n_components - 1:
    #         mu_smooth[i] = 0.99 * mu_new[i] + 0.01 * mu_new[i-1]
    #     else:
    #         mu_smooth[i] = 0.33 * mu_new[i-1] + 0.34 * mu_new[i] + 0.33 * mu_new[i+1]
    # mu_new = mu_smooth


    sigma_new = np.zeros((n_components, d, d))
    for i in range(n_components):
        data_centered = data - mu[i]
        sigma_tmp = np.einsum("ijk, ikl -> ijl", np.expand_dims(data_centered, 2), np.expand_dims(data_centered, 1))
        sigma_tmp = post_prob[:,i] * np.reshape(sigma_tmp, (N, d*d)).transpose()
        sigma_tmp = np.reshape(sigma_tmp.transpose(), (N, d, d))
        sigma_new[i] = np.sum(sigma_tmp, axis=0) / np.sum(post_prob[:,i])
        if mode == 'skeleton':
            sigma_new[i] = 0.5 * sigma_new[i] + 0.5 * 20 *np.eye(2)
        elif mode == 'seg':
            sigma_new[i] = 0.9 * sigma_new[i] + 0.1 * 20 *np.eye(2)
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


def em_mix(data, n_components, mu, mu_class_idx, n_stroke, canvas_size, mode):
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
    for em_iter in range(50):
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

        for idx_stroke in range(n_stroke): 
            plt.scatter(_mu[mu_class_idx[idx_stroke]][:, 0], _mu[mu_class_idx[idx_stroke]][:, 1], s=5)
        plt.scatter(data[:, 0], data[:, 1], s=0.5)
        for k in range(_mu.shape[0]):
            for i in range(len(mu_class_idx)):
                try:
                    mu_class_idx[i].index(k)
                except:
                    continue
                else:
                    color_idx = i
            plot_ellipses(mean=_mu[k], cov=sigma[k], ax=ax, alpha=0.1, color_idx=color_idx)
        if mode == 'skeleton':
            plt.savefig(f'./example_results/strokes/process_of_extract_SONG/process_01_skeleton/em_iter-{em_iter}.jpg')
        elif mode == 'seg':
            plt.savefig(f'./example_results/strokes/process_of_extract_SONG/process_02_segment/em_iter-{em_iter}.jpg')
        plt.close()

        mu_new, sigma_new, prior_new = one_EM_iteration(data, n_components, _mu, sigma, prior, n_stroke, mu_class_idx, mode, mu_init)
        
        norm_mu_diff = np.linalg.norm(mu_new - _mu) / _mu.shape[0]
        norm_sigma_diff = np.linalg.norm(sigma_new - sigma) / sigma.shape[0]
        
        print(f"diff_mu   :{norm_mu_diff}")
        print(f"diff_sigma:{norm_sigma_diff}")

        _mu = mu_new
        sigma = sigma_new
        prior = prior_new
        print(f'time{em_iter}:{time.time() - start}')

        stop_flag = (norm_mu_diff < 0.015) and (norm_sigma_diff < 0.04)
        if stop_flag == True:
            break

    return _mu, sigma, prior


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


def strokepoints_to_new_mu(stroke_points, nf_mu_sparsity):
    n_stroke = len(stroke_points)

    for idx_stroke in range(n_stroke):
        sparse_index = np.random.choice(
            np.arange(stroke_points[idx_stroke].shape[0]),
            size=int(stroke_points[idx_stroke].shape[0] / nf_mu_sparsity),
        )
        sparse_index = np.sort(sparse_index)
        stroke_points[idx_stroke] = stroke_points[idx_stroke][sparse_index]

    mu_class_idx = []
    for idx_stroke in range(n_stroke):
        if idx_stroke == 0:
            mu = stroke_points[idx_stroke]
            n_components = stroke_points[idx_stroke].shape[0]
            mu_class_idx.append(
                np.ndarray.tolist(
                    np.arange(0, stroke_points[idx_stroke].shape[0])
                )
            )
        else:
            mu = np.concatenate((mu, stroke_points[idx_stroke]), axis=0)
            n_components = n_components + stroke_points[idx_stroke].shape[0]
            mu_class_idx.append(
                np.ndarray.tolist(
                    np.arange(mu_class_idx[-1][-1] + 1, mu_class_idx[-1][-1] + 1 + stroke_points[idx_stroke].shape[0])
                )
            )
    return mu, n_components, n_stroke, mu_class_idx

# plot stroke images
def plot_stroke(data_idx, stroke_points, n_stroke, canvas_size):
    for idx_stroke in range(n_stroke):
        # get centroi
        centroi_x = np.sum(stroke_points[idx_stroke][:, 0]) / stroke_points[idx_stroke][:, 0].shape[0]
        centroi_y = np.sum(stroke_points[idx_stroke][:, 1]) / stroke_points[idx_stroke][:, 1].shape[0]
        x = stroke_points[idx_stroke][:, 0] - centroi_x + (canvas_size / 2)
        y = stroke_points[idx_stroke][:, 1] - centroi_y + (canvas_size / 2)

        if idx_stroke == 0:
            with open(f'./example_results/strokes/strokes_SONG/SONG_unicode{data_idx}.csv', 'w') as f:
                f.write(f"{centroi_x/canvas_size:.4f},{centroi_y/canvas_size:.4f},\n")
        else:
            with open(f'./example_results/strokes/strokes_SONG/SONG_unicode{data_idx}.csv', 'a') as f:
                f.write(f"{centroi_x/canvas_size:.4f},{centroi_y/canvas_size:.4f},\n")

        plt.figure(figsize=(3.05, 3.075), facecolor='black')
        ax = plt.gca()
        plt.xlim(0, canvas_size)
        plt.ylim(0, canvas_size)
        ax.invert_yaxis()
        plt.axis('off')
        plt.scatter(x, y, s=2, c='white')
        

        plt.savefig(f'./example_results/strokes/strokes_SONG/SONG_unicode{data_idx}_stroke{idx_stroke}.jpg', bbox_inches='tight', dpi=100)


def main(args):
    data_idx_start = args.unicode
    data_idx_end = data_idx_start
    for data_idx in range(data_idx_start, data_idx_end + 1):


        # config
        canvas_size = 360
        skeletion_em_data_sparsity = 1
        em_data_sparsity = 10
        mu_sparsity = 5
        nf_mu_sparsity = 2
        threshold = 0.3

        # get pixel coordinates
        # of skeleton of SONG
        pixel_full_x, pixel_full_y, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom = \
            get_pixel_coordinates(data_idx, canvas_size, font='SONG', skeletonize=True)
        sparse_index = np.random.choice(
            np.arange(pixel_full_x.shape[0]),
            size=int(pixel_full_x.shape[0] / skeletion_em_data_sparsity),
        )
        sparse_index = np.sort(sparse_index)
        pixel_x = pixel_full_x[sparse_index]
        pixel_y = pixel_full_y[sparse_index]
   

        # get mu, (of KAITI font)
        #     number of Gaussian Mixture Model compoments(num of mu),
        #     number of stroke, 
        #     mu each stroke's indices
        mu, n_components, n_stroke, mu_class_idx = get_mu_n(data_idx, mu_sparsity, font_lim_left, font_lim_right, font_lim_top, font_lim_bottom)

        # EM
        data_em = np.concatenate(
            (np.expand_dims(pixel_x, 1), np.expand_dims(pixel_y, 1)),
            axis=1,
            )
        mu_song, sigma, prior = em_mix(data=data_em, n_components=n_components, mu=mu, mu_class_idx=mu_class_idx,
                                n_stroke=n_stroke, canvas_size=canvas_size, mode='skeleton')


        # stroke segmentation
        # segment skeleton as the mu of the new font
        stroke_points = get_stroke_points(pixel_full_x, pixel_full_y, n_components, n_stroke, mu_song, mu_class_idx, sigma, prior, threshold)
        # nf short for new font
        nf_mu, nf_n_components, nf_n_stroke, nf_mu_class_idx = strokepoints_to_new_mu(stroke_points, nf_mu_sparsity)


        # get pixel coordinates
        # of new font
        nf_pixel_full_x, nf_pixel_full_y, _, _, _, _ \
            = get_pixel_coordinates(data_idx, canvas_size, font='SONG', skeletonize=False)
        nf_sparse_index = np.random.choice(
            np.arange(nf_pixel_full_x.shape[0]),
            size=int(nf_pixel_full_y.shape[0]/em_data_sparsity),
        )
        nf_sparse_index = np.sort(nf_sparse_index)
        nf_pixel_x = nf_pixel_full_x[nf_sparse_index]
        nf_pixel_y = nf_pixel_full_y[nf_sparse_index]

        # EM
        nf_data_em = np.concatenate(
            (np.expand_dims(nf_pixel_x, 1), np.expand_dims(nf_pixel_y, 1)),
            axis=1,
            )
        nf_mu, nf_sigma, nf_prior = em_mix(data=nf_data_em, n_components=nf_n_components, mu=nf_mu, mu_class_idx=nf_mu_class_idx,
                                n_stroke=nf_n_stroke, canvas_size=canvas_size, mode='seg')

        # stroke segmentation
        nf_stroke_points = get_stroke_points(nf_pixel_full_x, nf_pixel_full_y, nf_n_components, nf_n_stroke, nf_mu, nf_mu_class_idx, nf_sigma, nf_prior, threshold)

        # plot stroke images
        plot_stroke(data_idx, nf_stroke_points, nf_n_stroke, canvas_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unicode', type=int)
    args = parser.parse_args()

    if not os.path.exists('./example_results/strokes/strokes_SONG'):
        os.system('mkdir -p ./example_results/strokes/strokes_SONG')
    if not os.path.exists('./example_results/strokes/process_of_extract_SONG/process_01_skeleton/'):
        os.system('mkdir -p ./example_results/strokes/process_of_extract_SONG/process_01_skeleton/')
    if not os.path.exists('./example_results/strokes/process_of_extract_SONG/process_02_segment'):
        os.system('mkdir -p ./example_results/strokes/process_of_extract_SONG/process_02_segment')

    main(args)