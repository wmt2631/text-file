import matplotlib.pyplot as plt
import numpy as np


'''x = np.linspace(-3, 3, 50)
y = 2*x + 1
plt.figure(num=1, figsize=(8, 5),)
plt.plot(x, y,)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = 2*x0 + 1

plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5) #做线
plt.scatter([x0, ], [y0, ], s=50, color='b') #标注这个点


#plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),textcoords='offset points', fontsize=16,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2")) #标注
#plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',fontdict={'size': 16, 'color': 'r'})'''

#散点图
'''n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X)#颜色

plt.scatter(X,Y,s=75,c=T,alpha=0.5)


plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.xticks(())
plt.yticks(())'''

#树状图
'''n = 12
X = np.arange(n)

Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x, y in zip(X, Y1):
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')

plt.xlim(-.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())'''

#等高线
'''def f(x,y):
	return(1-x/2 + x**5+y**3)*np.exp(-x**2-y**2)#算高度

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot)#颜色
C = plt.contour(X,Y,f(X,Y),8,colors='black',linewidth=5)#划线
#8 分8次
plt.clabel(C,inline=True,fontsize=10)

plt.xticks()
plt.yticks() '''

#图像
'''a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

plt.imshow(a,interpolation='nearest',cmap='bone',origin='lower')
plt.colorbar()'''


#3D
'''from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z=np.sin(R)

ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))

ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
ax.set_zlim(-2,2)'''

#一屏多图
'''plt.figure()

plt.subplot(2,1,1)#两行两列第一个图
plt.plot([0,1],[0,1])

plt.subplot(2,3,4)#两行两列二
plt.plot([0,1],[0,2])

plt.subplot(2,3,5)#两行两列三
plt.plot([0,1],[0,3])

plt.subplot(2,3,6)#两行两列四
plt.plot([0,1],[0,4])'''


import matplotlib.gridspec as gridspec
#method1
'''import matplotlib.gridspec as gridspec
plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)#分几块 初始块 占几行几列
ax1.plot([1,2],[1,2])
ax1.set_title('ax1_title')
ax2 = plt.subplot2grid((3,3),(1,0),colspan=2)
ax3 = plt.subplot2grid((3,3),(1,2),rowspan=1)
ax4 = plt.subplot2grid((3,3),(2,0))
ax5 = plt.subplot2grid((3,3),(2,1))'''

#method2
'''
import matplotlib.gridspec as gridspec
plt.figure()
gs = gridspec.GridSpec(3,3)
ax1 = plt.subplot(gs[0,:])
ax2 = plt.subplot(gs[1,:2])
ax3 = plt.subplot(gs[1:,2])
ax4 = plt.subplot(gs[-1,0])
ax4 = plt.subplot(gs[-1,-2])'''

#method3
'''f,((ax11,ax12),(ax21,ax22))=plt.subplots(2,2,sharex=True,sharey=True)
ax11.scatter([1,2],[1,2])#散点图

plt.tight_layout()'''


#图中图 在同一个figure中
'''import matplotlib.gridspec as gridspec
fig = plt.figure()
x = [1,2,3,4,5,6,7]
y = [1,3,4,2,5,8,6]

left,bottom,widtn,height = 0.1,0.1,0.8,0.8
ax1 = fig.add_axes([left,bottom,widtn,height])
ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

left,bottom,widtn,height = 0.2,0.6,0.25,0.25
ax2 = fig.add_axes([left,bottom,widtn,height])
ax2.plot(x,y,'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title inside 1')

plt.axes([.6,0.2,0.25,0.25])
plt.plot(y[::-1],x,'g')
plt.xlabel('x')
plt.ylabel('y')'''

#两个坐标轴
'''x = np.arange(0,10,0.1)
y1 = 0.05*x**2
y2 = -1*y1

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()#反向
ax1.plot(x,y1,'g-')
ax2.plot(x,y2,'b--')

ax1.set_xlabel('sa')
ax1.set_ylabel('Y1',color='g')
ax2.set_ylabel('Y2',color='b')'''

#动画
from matplotlib import pyplot as plt
from matplotlib import animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))
def animate(i):
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,
def init():
    line.set_ydata(np.sin(x))
    return line,  
ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,interval=20, blit=False)


plt.show()







