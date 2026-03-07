# infer_one.py
import os
import numpy as np
import torch
from scipy.io import loadmat, savemat

from model import UNet3D  # your model.py

def norm01(a, eps=1e-6):
    a = a.astype(np.float32)
    amin, amax = float(a.min()), float(a.max())
    return (a - amin) / (amax - amin + eps)

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def main(
    mat_path="data_one_patient/patient001.mat",
    ckpt_path="checkpoints/best_unet3d_spect2ct.pt",
    out_path="outputs/patient001_pred_mu.mat",
    photo_key="photopeak",
    scatter_key="scatter",
    normalize_inputs=True,   # MUST match training
    clip_mu=True,
    mu_min=0.0,
    mu_max=0.30,             # adjust to your dataset range
):
    os.makedirs("outputs", exist_ok=True)

    device = pick_device()
    print("Using device:", device)

    d = loadmat(mat_path)
    photo = np.asarray(d[photo_key], dtype=np.float32)   # (H,W,D)
    scat  = np.asarray(d[scatter_key], dtype=np.float32)

    if photo.shape != scat.shape:
        raise ValueError(f"Shape mismatch: photo {photo.shape}, scatter {scat.shape}")

    # --- preprocessing must match training ---
    if normalize_inputs:
        photo_n = norm01(photo)
        scat_n  = norm01(scat)
    else:
        photo_n, scat_n = photo, scat

    # x: (B,C,D,H,W)
    x = np.stack([photo_n, scat_n], axis=0)     # (2,H,W,D)
    x = np.transpose(x, (0, 3, 1, 2))           # (2,D,H,W)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    model = UNet3D(in_ch=2, out_ch=1, base=16).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    pred = model(x)  # (1,1,D,H,W)
    pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy()  # (D,H,W)
    pred_hwd = np.transpose(pred, (1, 2, 0))                  # (H,W,D)

    if clip_mu:
        pred_hwd = np.clip(pred_hwd, mu_min, mu_max)

    # Save µ-map prediction
    savemat(out_path, {
        "pred_mu": pred_hwd.astype(np.float32),
        "photopeak": photo.astype(np.float32),
        "scatter": scat.astype(np.float32),
    })
    print("Saved:", out_path)

if __name__ == "__main__":
    main()

d = load('outputs/patient001_pred_mu.mat');
mu = d.pred_mu;   % 128x128x28

fprintf('mu range: min=%g max=%g\n', min(mu(:)), max(mu(:)));

figure;
for k=1:size(mu,3)
    imagesc(mu(:,:,k), [0 0.25]); axis image off; colormap gray; colorbar;
    title(sprintf('Predicted \\mu-map slice %d', k));
    pause(0.1);
end
