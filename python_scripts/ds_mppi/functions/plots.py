import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def init_robot_plot(links, xmin, xmax, ymin, ymax):
    # Initialize the robot plot
    plt.ion()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_aspect('equal')
    line_h, = plt.plot([], [], 'o-', color=[0, 0.4470, 0.7410], linewidth=3, markersize=5)
    plt.title('Planar Robot')
    plt.show()
    return line_h


def init_jpos_plot(xmin, xmax, ymin, ymax):
    # Initialize the joint position plot
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('First Joint, radians')
    ax.set_ylabel('Second Joint, radians')
    ax.set_aspect('equal')
    rob, = plt.plot([], [], 'o', color=[0, 0.4470, 0.7410], markersize=7)
    plt.ion()
    plt.title('Joint Space')
    plt.show()
    return rob

def init_kernel_means(n_kernel_max):
    c_h = []
    for i in range(n_kernel_max):
        # tmp, = plt.plot([], [], '-o', color=[0, 0.60, 0.90], markersize=2, linewidth=.5)
        tmp, = plt.plot([], [], '-o', color='g', markersize=2, linewidth=.5)

        c_h.append(tmp)
    return c_h

def upd_jpos_plot(jpos, ln):
    # Update the joint position plot
    ln.set_data(jpos[0], jpos[1])
    plt.draw()
    return 0


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

def upd_toy_h(coord, ln):
    ln.set_data(coord[0], coord[1])
    plt.draw()
    return 0

def plot_circ(c, r):
    # Plot a circle
    fig = plt.figure(1)
    ax = fig.get_axes()[0]
    circ = plt.Circle(c[0:2], r, color='r', fill=False, linewidth=2)
    ax.add_patch(circ)
    return circ


def plot_obs_init(obstacles):
    o_h = []
    for obstacle in obstacles:
        o_h.append(plot_circ(obstacle[0:2], obstacle[-1]))
    return o_h


def plot_obs_update(o_h, obstacles):
    for i, obstacle in enumerate(obstacles):
        o_h[i].center = obstacle[0:2]



def init_robot_plot3d(xmin, xmax, ymin, ymax, zmin, zmax, width, color, markersize):
    # Initialize the robot plot
    plt.ion()
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('auto')
    line_h, = ax.plot3D([], [], [], 'o-', color=color, linewidth=width, markersize=markersize)
    plt.show()
    return line_h

def init_line3d(width, color, markersize):
    # Initialize the robot plot
    fig = plt.figure(1)
    ax = fig.get_axes()[0]
    line_h, = ax.plot3D([], [], [], 'o-', color=color, linewidth=width, markersize=markersize)
    plt.show()
    return line_h

def upd_r_h3d(links, ln):
    # Update the robot plot in 3d
    x_data = [0,links[0][0, 0]]
    y_data = [0,links[0][0, 1]]
    z_data = [0,links[0][0, 2]]
    # add link end points to plot
    for link in links:
        x_data.append(link[-1, 0])
        y_data.append(link[-1, 1])
        z_data.append(link[-1, 2])
    ln.set_xdata(x_data)
    ln.set_ydata(y_data)
    ln.set_3d_properties(z_data)
    plt.draw()
    return 0


def init_toy_plot(xmin, xmax, ymin, ymax):
    # Initialize the robot plot
    plt.ion()
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    line_h, = plt.plot([], [], '*', color=[0, 0.4470, 0.7410], linewidth=5, markersize=5)
    plt.show()
    return line_h
