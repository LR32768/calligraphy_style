# 引用所需要的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl


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
    # 是否画出特征向量
    if eigv:
        # print "type(v): ", type(v)
        if arrow_color_list is None:
            arrow_color_list = [color for i in range(v.shape[0])]
        for i in range(v.shape[0]):
            v_i = v[:, i]
            scale_variable = np.sqrt(s) * sqrt_lambda[i]
            # 绘制箭头
            """
            ax.arrow(x, y, dx, dy,    # (x, y)为箭头起始坐标，(dx, dy)为偏移量
                     width,    # 箭头尾部线段宽度
                     length_includes_head,    # 长度是否包含箭头
                     head_width,    # 箭头宽度
                     head_length,    # 箭头长度
                     color,    # 箭头颜色
                     )
            """
            ax.arrow(mean[0], mean[1], scale_variable*v_i[0], scale_variable * v_i[1], 
                     width=0.05, 
                     length_includes_head=True, 
                     head_width=0.2, 
                     head_length=0.3,
                     color=arrow_color_list[i])


# 获取直线的数组
def line_space(B):
    x = np.linspace(0, 10, 100)
    return x, x + B


# 这里B就是frame
def update(B):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    x, y = line_space(B)
    line.set_data(x, y)
    breakpoint()
    return line



# 定义画布
fig, ax = plt.subplots()
line, = ax.plot([], [])  # 返回的第一个值是update函数需要改变的

ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), interval=50)
ani.save(f'./extract_stroke/gif.gif', writer='imagemagick', fps=10)
plt.show()