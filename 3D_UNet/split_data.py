import os
import shutil
import random

def split_dataset(source_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    source_dir = os.path.abspath(source_dir)
    train_dir = os.path.join(source_dir, "TrainMat")
    val_dir   = os.path.join(source_dir, "ValdMat")
    test_dir  = os.path.join(source_dir, "TestMat")

    # Create directories
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Identify patient files (P_xxx.mat)
    all_files = sorted([
        f for f in os.listdir(source_dir) 
        if f.startswith("P_") and f.endswith(".mat") and os.path.isfile(os.path.join(source_dir, f))
    ])

    if not all_files:
        print(f"No files found to split in: {source_dir}")
        print("Note: If folders already exist, they might be inside them.")
        return

    print(f"Found {len(all_files)} files. Splitting {int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}...")

    random.seed(42) 
    random.shuffle(all_files)

    total = len(all_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_files = all_files[:train_count]
    val_files   = all_files[train_count : train_count + val_count]
    test_files  = all_files[train_count + val_count:]

    def move_files(file_list, target_path):
        for f in file_list:
            src = os.path.join(source_dir, f)
            dst = os.path.join(target_path, f)
            # Check if file exists in destination to avoid shutil.Error
            if os.path.exists(dst):
                os.remove(dst) 
            shutil.move(src, dst)

    move_files(train_files, train_dir)
    move_files(val_files, val_dir)
    move_files(test_files, test_dir)
    
    print(f" Split Complete: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

def merge_back(source_dir):
    source_dir = os.path.abspath(source_dir)
    subfolders = ["TrainMat", "ValdMat", "TestMat"]
    
    for sub in subfolders:
        path = os.path.join(source_dir, sub)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            for f in files:
                src = os.path.join(path, f)
                dst = os.path.join(source_dir, f)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
            if files:
                print(f"Moved {len(files)} files back from {sub}")

if __name__ == "__main__":
    # Script location: /3D_UNet/split_data.py
    # Data location: /data/
    DATA_PATH = "../data" 
    split_dataset(DATA_PATH)
    # merge_back(DATA_PATH)
