import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_kitti_poses(path):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    N = raw.shape[0]
    if raw.shape[1] != 12:
        raise ValueError(f"Expected KITTI poses with 12 columns, got {raw.shape[1]}")

    traj = np.zeros((N, 4, 4), dtype=np.float64)
    traj[:, 3, 3] = 1.0

    # raw[i] : 12 -> 3x4
    for i in range(N):
        traj[i, :3, :4] = raw[i].reshape(3, 4)

    return traj


def load_result(path):
    raw = np.loadtxt(path, delimiter=" ")
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 3:
        raw = raw.reshape(-1, 3)
    return raw


def loop_vis(traj, result, sim_th=0.1):
    N = traj.shape[0]
    x = traj[:, 0, 3]
    y = traj[:, 2, 3] 
    z = np.arange(N, dtype=np.float64) * 0.0001
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='black', linewidth=1, label="Trajectory")

    for row in result:
        src = int(row[0])
        tgt = int(row[1])
        sim = float(row[2])

        if tgt < 0:
            continue
        if sim < sim_th:
            continue
        if not (0 <= src < N and 0 <= tgt < N):
            continue
        if tgt >= src:  
            continue

        ax.plot([x[src], x[tgt]],
                [y[src], y[tgt]],
                [z[src], z[tgt]],
                color='green',
                linewidth=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    x_mid = 0.5 * (x.min() + x.max())
    y_mid = 0.5 * (y.min() + y.max())
    xy_range = max(x.max() - x.min(), y.max() - y.min()) * 0.5
    if xy_range == 0:
        xy_range = 1.0

    ax.set_xlim(x_mid - xy_range, x_mid + xy_range)
    ax.set_ylim(y_mid - xy_range, y_mid + xy_range)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    traj_path = "/root/FLASH/dataset/SemanticKITTI/00/00.txt"
    result_path = "/root/FLASH/module/src/result.txt"

    traj = load_kitti_poses(traj_path)
    result_arr = load_result(result_path)

    loop_vis(traj, result_arr, sim_th=0.1)
