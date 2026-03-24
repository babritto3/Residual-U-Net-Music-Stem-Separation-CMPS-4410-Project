import torch
import os
import torch_directml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from Model import UNet 
from dataset import MusDB_Dataset 

# --- Ensure path is accessible for Windows DML libraries ---
import os
os.environ["PATH"] += os.pathsep + os.getcwd()

def check_accuracy(loader, model, loss_fn, device):
    """Calculates the average loss on the validation set."""
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            print(f"   > Validating batch {i+1}/{len(loader)}...", end='\r')
    
    print() # Newline after validation printing
    avg_loss = total_loss / len(loader)
    model.train()
    return avg_loss

# --- MAIN BLOCK ---
if __name__ == '__main__':
    
    DEVICE = torch_directml.device() if torch_directml.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- HYPERPARAMETERS ---
    LEARNING_RATE = 1e-4      # Fine-tuning speed
    BATCH_SIZE = 2            # Max physical batch size
    ACCUMULATION_STEPS = 4    # Effective Batch Size 8
    NUM_EPOCHS = 200          
    
    # Path to your dataset
    MUSDB_ROOT = "C:\\Users\\babri\\Documents\\CMPS 4410 Final Project\\Musdb datatset\\musdb18" 

    # 1. Dataset and Loaders
    train_dataset = MusDB_Dataset(MUSDB_ROOT, is_train=True, chunk_duration=5)
    test_dataset = MusDB_Dataset(MUSDB_ROOT, is_train=False, chunk_duration=5)
    
    # Use a small test subset for faster validation checking
    test_subset = Subset(test_dataset, list(range(10))) 

    # num_workers=4 helps keep the GPU fed
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = UNet(num_classes=8).to(DEVICE)
    
    # --- LOSS FUNCTION (Standard MSE) ---
    loss_fn = nn.MSELoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
        threshold=0.005, threshold_mode='abs'
    )
    # --- LOADING BLOCK ---
    LOAD_EXISTING_MODEL = True
    
    # Initialize best loss to infinity so we can track improvements in this session
    best_test_loss = float("0.1731") 

    if LOAD_EXISTING_MODEL:
        print("Attempting to load best_model.pth...")
        
        if not os.path.exists("best_model.pth"):
            raise FileNotFoundError("CRITICAL: 'best_model.pth' was not found! Training stopped.")
            
        # Load weights
        state_dict = torch.load("best_model.pth", map_location=DEVICE, weights_only=False)
        model.load_state_dict(state_dict, strict = False)
        print("✅ SUCCESS: Loaded weights from (best_model.pth)")
    else:
        print("⚠️ WARNING: Starting training from SCRATCH.")

    print(f"Starting with Effective Batch Size {BATCH_SIZE * ACCUMULATION_STEPS}...")

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        model.train()
        
        optimizer.zero_grad() 

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            
            # --- VIRTUAL BATCH / GRADIENT ACCUMULATION ---
            loss = loss / ACCUMULATION_STEPS 
            loss.backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()      
                optimizer.zero_grad() 
            
            running_loss += (loss.item() * ACCUMULATION_STEPS)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}: Batch {batch_idx+1}/{len(train_loader)} processed. Current MSE: {loss.item() * ACCUMULATION_STEPS:.4f}", end='\r')

        print() 
        epoch_loss = running_loss / len(train_loader)
        
        print("Calculating test loss...")
        test_loss = check_accuracy(test_loader, model, loss_fn, DEVICE)
        
        # Manual Learning Rate Print
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train MSE: {epoch_loss:.4f}, Test MSE: {test_loss:.4f}, LR: {current_lr:.6f}")

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved! (Loss: {best_test_loss:.4f})")

        torch.save(model.state_dict(), "latest_model.pth")

    print("Finished Training!")