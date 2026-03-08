import os
import torch
import numpy as np 
import time  
from torch.utils.data import DataLoader
from scipy.io import savemat

from dataset import SPECT2CTDataset
from model import UNet3D

def pick_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(test_dir="../data/TestMat", 
             ckpt_path="checkpoints/best_model.pt",
             out_dir="output/test_result"): 
    
    device = pick_device()
    print(f"Evaluating on: {device}")
    os.makedirs(out_dir, exist_ok=True)

    # Load Dataset
    ds = SPECT2CTDataset(test_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    # Load Model (base=32 for L40S)
    model = UNet3D(in_ch=2, out_ch=1, base=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    print(f"Found {len(ds)} test cases. Starting inference...")
    
    inference_times = []
    total_start_time = time.time()

    for bi, (x, y) in enumerate(loader):
        original_path = ds.files[bi]
        patient_id = os.path.basename(original_path).replace(".mat", "")

        start_tick = time.time()

        # Inference
        x = x.to(device)
        pred = model(x)
        
        # Ensure GPU finishes work before stopping the clock
        if device.type == 'cuda':
            torch.cuda.synchronize()         
        end_tick = time.time()

        duration = end_tick - start_tick
        inference_times.append(duration)

        # Transpose to MATLAB format (H, W, D)
        p_vol = pred[0, 0].cpu().numpy().transpose(1, 2, 0)
        t_vol = y[0, 0].cpu().numpy().transpose(1, 2, 0)

        # Save to MAT
        save_path = os.path.join(out_dir, f"{patient_id}_unet.mat")
        savemat(save_path, {
            'pred_attnmap': p_vol,
            'target_attnmap': t_vol,
            'inference_time_sec': duration # Optional: save time inside the .mat
        })

        print(f"Processed: {patient_id} | Time: {duration:.4f}s")

    total_duration = time.time() - total_start_time
    avg_time = np.mean(inference_times)
    
    print("-" * 30)
    print(f"Evaluation Complete!")
    print(f"Total Time for {len(ds)} patients: {total_duration:.2f}s")
    print(f"Average Time per Synthesis: {avg_time:.4f}s")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    evaluate()
