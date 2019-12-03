import matplotlib.pyplot as plt
import datasets.dataset_provider as data_provider

import numpy as np

data_iris = data_provider.getIris()
y = data_iris.iloc[:, 0].values


def get_random_list(start=2, end=5):
    return np.random.randint(start, end, 5, dtype='int8')


def get_sum_of_column():
    q, w, e, r = 0, 0, 0, 0
    for col in data_iris.iloc[:, 0:-1].values:
        q += col[0]
        w += col[1]
        e += col[2]
        r += col[3]
    return [q, w, e, r]


def lines():
    plt.plot([1, 2, 3], [2, 3, 4], label='Some line')
    plt.plot([2, 3, 4], [2, 3, 4], label='Second line')


def hist():
    plt.hist(y, histtype='bar', rwidth=0.8)


def bar():
    plt.bar(*(get_random_list() for _ in range(2)), label='Another Bars', color='r')
    plt.bar([x for x in range(y.size)], y, label='Bars')


def scatter():
    plt.scatter([x for x in range(y.size)], y)


def stack_plot():
    labels = data_iris.columns[:-1]
    plt.plot([], [], color='c', label=labels[0], linewidth=6)
    plt.plot([], [], color='g', label=labels[1], linewidth=6)
    plt.plot([], [], color='y', label=labels[2], linewidth=6)
    plt.plot([], [], color='m', label=labels[3], linewidth=6)
    plt.stackplot(labels, *get_sum_of_column(), colors=['c', 'g', 'y', 'm'])


def pie():
    plt.pie([int(x) for x in get_sum_of_column()], labels=data_iris.columns[:-1], colors=['c', 'g', 'y', 'm'],
            autopct='%1.1f%%')


def draw_plot(tiny=False):
    if not tiny:
        plt.xlabel('X - axis')
        plt.ylabel('Y - axis')
        plt.legend()  # show labels

    plt.title('Simple plot\nWith title')
    plt.show()


if __name__ == '__main__':
    # lines()
    # bar()
    # hist()
    # scatter()
    # stack_plot()
    pie()

    draw_plot(True)
