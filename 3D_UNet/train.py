import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from dataset import SPECT2CTDataset
from model import UNet3D
from utils import EarlyStopping

# Setup Directories
os.makedirs("checkpoints", exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():
    set_seed(42) # Ensure training is reproducible
    
    # Use CUDA 0 (NVIDIA L40S)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (NVIDIA L40S)")

    # Data Loading 
    train_ds = SPECT2CTDataset("../data/TrainMat")
    val_ds   = SPECT2CTDataset("../data/ValdMat")

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, 
        num_workers=8, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=16, shuffle=False, 
        num_workers=8, pin_memory=True
    )

    # UNet_Model - base=32
    model = UNet3D(in_ch=2, out_ch=1, base=32).to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)
    stopper = EarlyStopping(patience=50)

    best_val = float("inf")

    print(f"Starting training with {len(train_ds)} cases...")

    for epoch in range(1, 1001):       
        # Training Phase
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # Early Stopping and Checkpointing
        stop_now, is_best = stopper.step(val_loss)
        
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} *Best*")
        else:
            if epoch % 5 == 0: # Print more frequently now that you have more data
                print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if stop_now:
            print(f"Stopping at epoch {epoch}. Best Val Loss: {best_val:.6f}")
            break

if __name__ == "__main__":
    train()
