import os
import shutil
import random

def split_dataset(source_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):

    # Setup paths
    train_dir = os.path.join(source_dir, "TrainMat")
    val_dir   = os.path.join(source_dir, "ValdMat")
    test_dir  = os.path.join(source_dir, "TestMat")

    # Create directories if they don't exist
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Get all .mat files that follow the P_xxx.mat pattern
    all_files = sorted([
        f for f in os.listdir(source_dir) 
        if f.startswith("P_") and f.endswith(".mat")
    ])

    if not all_files:
        print(f"No patient files found in {source_dir}")
        return

    # 3. Shuffle files to ensure random distribution
    random.seed(42) # For reproducibility
    random.shuffle(all_files)

    # 4. Calculate split indices
    total = len(all_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_files = all_files[:train_count]
    val_files   = all_files[train_count : train_count + val_count]
    test_files  = all_files[train_count + val_count:]

    # move files
    def move_files(file_list, target_path):
        for f in file_list:
            src = os.path.join(source_dir, f)
            dst = os.path.join(target_path, f)
            shutil.move(src, dst)
        print(f"Moved {len(file_list)} files to {target_path}")

    # Execute move
    move_files(train_files, train_dir)
    move_files(val_files, val_dir)
    move_files(test_files, test_dir)

    print("\n Splitting complete!")

if __name__ == "__main__":
    split_dataset("./")
