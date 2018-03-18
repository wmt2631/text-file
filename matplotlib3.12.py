import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x**2

plt.figure()
plt.plot(x, y2)
plt.plot(x, y1, linestyle='--', color='red')

plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('i an x')
plt.ylabel('i an y')

new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)
# set tick labels
plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$',
                                     r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])



# 该坐标轴
ax = plt.gca()
ax.spines['right'].set_color('none')#右边边框无色
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')#用底边框代替x轴
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))#x设置位置
ax.spines['left'].set_position(('data', 0))

# 加图例
l1, = plt.plot(x, y2, label='up')
l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='down')
plt.legend(handles=[l1, l2], labels=['aa'], loc='best')

plt.show()
