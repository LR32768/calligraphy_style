# 引用所需要的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定义画布
fig, ax = plt.subplots()
line, = ax.plot([], [])  # 返回的第一个值是update函数需要改变的


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
    return line

# 使用函数并保存保存会在下一篇文章讲
# 可以用plt.show()来代替一下
ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), interval=50)
# ani.save('move1.gif', writer='imagemagick', fps=10)
plt.show()