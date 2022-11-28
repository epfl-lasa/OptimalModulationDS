import matplotlib.pyplot as plt


def init_robot_plot(links, xmin, xmax, ymin, ymax):
    # Initialize the robot plot
    plt.ion()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    line_h, = plt.plot([], [], 'o-', color=[0, 0.4470, 0.7410], linewidth=3, markersize=5)
    plt.show()
    return line_h


def upd_r_h(links, ln):
    # Update the robot plot
    x_data = [links[0][0, 0]]
    y_data = [links[0][0, 1]]
    # add link end points to plot
    for link in links:
        x_data.append(link[-1, 0])
        y_data.append(link[-1, 1])
    ln.set_data(x_data, y_data)
    plt.draw()
    return 0


def plot_circ(c, r):
    # Plot a circle
    fig = plt.figure(1)
    ax = fig.get_axes()[0]
    circ = plt.Circle(c[0:2], r, color='r', fill=False)
    ax.add_patch(circ)
    return circ

