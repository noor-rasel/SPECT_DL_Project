import torch

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    Returns: (stop_triggered, is_best)
    """
    def __init__(self, patience=50):
        self.patience = patience
        self.best = float("inf")
        self.bad = 0

    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.bad = 0
            # stop_triggered = False, is_best = True
            return False, True 
        else:
            self.bad += 1
            # stop_triggered = (True if limit reached), is_best = False
            return self.bad >= self.patience, False