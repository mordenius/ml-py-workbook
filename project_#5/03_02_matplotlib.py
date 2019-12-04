import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')

fig = plt.figure()
axl = fig.add_subplot(1, 1, 1)


def animate(i):
    xs = np.random.random(5)
    ys = np.random.random(5)

    axl.clear()
    axl.plot(xs, ys)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
