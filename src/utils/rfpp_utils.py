import torch
import json
from torchvision.utils import save_image


def straightness(traj, mean=True):
    N = len(traj) - 1
    dt = 1 / N
    base = traj[0] - traj[-1]
    mse = []
    for i in range(1, len(traj)):
        v = (traj[i-1] - traj[i]) / dt
        if mean:
            # Average along the batch dimension
            mse.append(torch.mean((v - base) ** 2))
        else:
            # Average except the batch dimension
            if len(v.shape) == 2:
                mse.append(torch.mean((v - base) ** 2, dim=1))
            elif len(v.shape) == 4:
                mse.append(torch.mean((v - base) ** 2, dim=[1, 2, 3]))
    mse = torch.stack(mse)
    if mean:
        return torch.mean(mse)
    else:
        return torch.mean(mse, dim=0)


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_traj(traj, path):
    traj = torch.cat(traj, dim=3)
    traj = traj.permute(1, 0, 2, 3).contiguous().view(
        traj.shape[1], -1, traj.shape[3])
    save_image(traj * 0.5 + 0.5, path)
    print(f"Saved trajectory to {path}")
