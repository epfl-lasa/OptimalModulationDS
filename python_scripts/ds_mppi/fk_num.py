import torch
import numpy as np
from plots import *

@torch.jit.script
def dh_transform(q, d, theta, a, alpha):
    """
    Denavit-Hartenberg transformation matrix.
    """
    # Compute the transformation matrix
    sa = torch.sin(alpha)
    ca = torch.cos(alpha)
    sq = torch.sin(q+theta)
    cq = torch.cos(q+theta)
    # T = torch.tensor([
    #     [cq,        -sq,    0.0,      a],
    #     [sq * ca,   cq*ca,  -sa,    -d*sa],
    #     [sq * sa,   cq*sa,  ca,     d*ca],
    #     [0.0, 0.0, 0.0, 1.0]]).to(q.device, q.dtype)
    t1 = torch.hstack((cq, -sq, torch.tensor(0.0).to(q.device), a))
    t2 = torch.hstack((sq * ca, cq*ca, -sa, -d*sa))
    t3 = torch.hstack((sq * sa, cq*sa, ca, d*ca))
    t4 = torch.tensor((0.0, 0.0, 0.0, 1.0)).to(q.device)
    T  = torch.vstack((t1, t2, t3, t4))
    return T


@torch.jit.script
def dh_fk(q: torch.Tensor, dh_params: torch.Tensor):
    """
    Forward kinematics for a robot with Denavit-Hartenberg parameters.
    """
    # Initialize the transformation matrix
    T = [torch.eye(4).to(q.device)]
    # Loop through each joint
    for i in range(len(q)):
        # Extract the parameters for this joint
        d = dh_params[i, 0]
        theta = dh_params[i, 1]
        a = dh_params[i, 2]
        alpha = dh_params[i, 3]
        # Compute the transformation for this joint
        T_prev = T[-1]
        T.append(T_prev @ dh_transform(q[i], d, theta, a, alpha))
    return T


@torch.jit.script
def numeric_fk_model(q: torch.Tensor, dh_params: torch.Tensor, n_pts: int):
    """
    Caclulate positions of points on the robot arm.
    """
    # Compute the transformation matrices
    n_dof = len(q)
    P_arr = dh_fk(q, dh_params)
    robot = dict()
    links = []
    pts_int = []
    # Initialize the points array
    # Loop through each joint
    a = dh_params[:, 2]
    for i in range(n_dof):
        p0 = torch.tensor([0, 0, 0]).to(q.device)
        p1 = torch.hstack((a[i+1], torch.tensor([0.0, 0.0]).to(q.device)))
        lspan = torch.linspace(0, 1, n_pts).unsqueeze(1).to(q.device)
        v = torch.tile(p0, (n_pts, 1)) + torch.tile(p1, (n_pts, 1)) * lspan

        R = P_arr[i + 1][:3, :3]
        T = P_arr[i + 1][:3, 3]
        # Compute the position of the point for this link
        pts = (R @ v.transpose(0, 1)).transpose(0, 1) + T
        links.append(pts)
        pts_int.append(v)
    return links, pts_int


def dist_to_point(robot, y):
    """
    Calculate the distance between the robot links and a point in task space.
    """
    mindists = []
    minidxs = []
    res = dict()
    for link_pts in robot['links']:
        dist = np.linalg.norm(link_pts - y, 2, 1)
        minidx = np.argmin(dist)
        mindists.append(dist[minidx])
        minidxs.append(minidx)
    minidx = np.argmin(mindists)
    res['mindist'] = mindists[minidx]
    res['linkidx'] = minidx
    res['ptidx'] = minidxs[minidx]
    return res



def main():
    params = {'device': 'cpu', 'dtype': torch.float32}
    q = torch.tensor([0, 0]).to(**params)
    dh_a = torch.tensor([0, 3, 3])
    dh_alpha = dh_a*0
    dh_d = dh_a*0
    dh_theta = dh_a*0
    dh_params = (torch.vstack((dh_d, dh_theta, dh_a, dh_alpha)).T).to(**params)
    link_pts = numeric_fk_model(q, dh_params, 10)
    for i in range(300):
        link_pts, int_pts = numeric_fk_model(q, dh_params, 10)

if __name__ == '__main__':
    main()



