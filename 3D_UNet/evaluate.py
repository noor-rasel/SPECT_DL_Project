import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.io import savemat  

from dataset import SPECT2CTDataset  
from model import UNet3D

def pick_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(split_dir="../data_1/TestMat",  # UPDATED PATH
             ckpt_path="checkpoints/best_model.pt",
             batch_size=1, 
             save_plots=True,
             save_mats=True,
             out_dir="output/test_result"):
    
    device = pick_device()
    print(f"Evaluating on: {device}")
    
    # Create subdirectories for organized output
    plot_dir = os.path.join(out_dir, "plot")
    mat_dir = os.path.join(out_dir, "mat_file")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)

    # Dataset will now look into ../data_1/TestMat
    ds = SPECT2CTDataset(split_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = UNet3D(in_ch=2, out_ch=1, base=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    mse_sum, mae_sum, nvox_sum = 0.0, 0.0, 0

    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # Global Metrics
        mse_sum += F.mse_loss(pred, y, reduction="sum").item()
        mae_sum += F.l1_loss(pred, y, reduction="sum").item()
        nvox_sum += y.numel()

        # Extract 3D Volume (D, H, W) -> (H, W, D) for MATLAB
        p_vol = pred[0, 0].cpu().numpy().transpose(1, 2, 0)
        t_vol = y[0, 0].cpu().numpy().transpose(1, 2, 0)

        # 1. Save as .mat file
        if save_mats:
            mat_filename = os.path.join(mat_dir, f"P_{bi:03d}_pred.mat")
            savemat(mat_filename, {
                'pred_attnmap': p_vol,
                'target_attnmap': t_vol,
                'description': 'Generated via 3D U-Net SPECT2CT'
            })

        # 2. 3-Plane Visualization
        if save_plots:
            h_idx, w_idx, d_idx = np.array(p_vol.shape) // 2

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            plt.suptitle(f"Case {bi} Evaluation")

            planes = [
                (t_vol[:, :, d_idx], p_vol[:, :, d_idx], "Axial"),
                (t_vol[:, w_idx, :], p_vol[:, w_idx, :], "Sagittal"),
                (t_vol[h_idx, :, :], p_vol[h_idx, :, :], "Coronal")
            ]

            for i, (ground_truth, prediction, title) in enumerate(planes):
                axes[0, i].imshow(ground_truth, cmap='bone')
                axes[0, i].set_title(f"Target {title}")
                im = axes[1, i].imshow(prediction, cmap='bone')
                axes[1, i].set_title(f"Predicted {title}")
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

            plt.tight_layout()
            # Saved with P_ prefix to match your request
            plt.savefig(os.path.join(plot_dir, f"P_{bi:03d}.png"))
            plt.close()

    mse_mean = mse_sum / nvox_sum
    print(f"\nFinal Statistics:")
    print(f"RMSE: {np.sqrt(mse_mean):.6e}")
    print(f"MAE:  {mae_sum / nvox_sum:.6e}")
    print(f"MAT files saved to: {mat_dir}")

if __name__ == "__main__":
    evaluate()
