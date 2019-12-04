import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from matplotlib import style
import numpy as np

# style.use('fivethirtyeight')

fig = plt.figure()
# axl = fig.add_subplot(2, 2, 1)
# axl = fig.add_subplot(2, 2, 2)
# axl = fig.add_subplot(2, 1, 2)

axl = plt.subplot2grid((6,1), (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((6,1), (1,0), rowspan=4, colspan=1)
ax3 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1)

def animate(i):
    xs = np.random.random(5)
    ys = np.random.random(5)

    axl.clear()
    axl.plot(xs, ys)

    ax2.plot(ys, xs)

    ax3.clear()
    ax3.plot(ys, xs)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
