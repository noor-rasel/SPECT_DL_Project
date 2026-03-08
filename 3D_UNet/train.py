import os
import torch
import random
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
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

def plot_history(history, save_path='Training_Status.png'):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss', color='tab:blue')
    ax1.plot(epochs, history['train_loss'], color='tab:blue', alpha=0.4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color='tab:blue', linewidth=2, label='Val Loss')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Learning Rate', color='tab:red')
    ax2.plot(epochs, history['lr'], color='tab:red', linestyle='--', label='LR')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Training/Validation Loss & Learning Rate Schedule')
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.savefig(save_path)
    plt.close()

def train():
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (NVIDIA L40S)")

    # Data
    train_ds = SPECT2CTDataset("../data/TrainMat")
    val_ds   = SPECT2CTDataset("../data/ValdMat")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    # Model & Optimization
    model = UNet3D(in_ch=2, out_ch=1, base=32).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)
    stopper = EarlyStopping(patience=50)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val = float("inf")
    
    total_start_time = time.time()
    epoch_times = []

    print(f"Starting training with {len(train_ds)} cases...")

    max_epochs = 1000
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        
        # --- Training ---
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

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)

        # Record metrics
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Timing math
        epoch_end = time.time()
        duration = epoch_end - epoch_start
        epoch_times.append(duration)
        avg_time = np.mean(epoch_times)
        
        # Estimating remaining time (to max_epochs)
        remaining_epochs = max_epochs - epoch
        eta_seconds = remaining_epochs * avg_time
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        # Console Output
        status = f"Epoch {epoch:03d} | Loss: {train_loss:.6e} | Val: {val_loss:.6e} | Time: {duration:.2f}s | ETA: {eta_str}"
        
        stop_now, is_best = stopper.step(val_loss)
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"{status} *Best*")
            plot_history(history)
        elif epoch % 5 == 0:
            print(status)
            plot_history(history)

        if stop_now:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    total_duration = time.time() - total_start_time
    total_str = str(datetime.timedelta(seconds=int(total_duration)))
    print(f"\nTraining Complete!")
    print(f"Total Training Time: {total_str}")
    print(f"Average Time per Epoch: {np.mean(epoch_times):.2f}s")
    plot_history(history)

if __name__ == "__main__":
    train()
