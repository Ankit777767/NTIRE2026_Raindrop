# import os
# import torch
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from tqdm import tqdm
# from torch.cuda.amp import GradScaler, autocast
# # Import our custom modules
# from data.dataset import NTIRE2026Dataset
# from models.restormer import Restormer
# from models.losses import CharbonnierLoss
# from utils.metrics import NTIREMetric

# # --- Configuration ---
# CONFIG = {
#     'EPOCHS': 100,
#     'BATCH_SIZE': 12,       # Decrease if OOM (Out of Memory)
#     'PATCH_SIZE': 128,     # Restormer uses 128 or 256
#     'LR': 2e-4,            # Learning Rate
#     'NUM_WORKERS': 4,      # Set to 0 on Windows if you get pickle errors
#     'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
#     'DATA_ROOT': '/media/admin1/DL/ankit/raindrop/ntire2026-dualfocus-raindrop/data/train', # CHANGE THIS to your actual path
#     'SAVE_DIR': './checkpoints',
#     'VAL_INTERVAL': 5      # Validate every 5 epochs
# }

# def train():
#     os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
    
#     # 1. Dataset & Dataloaders
#     train_dataset = NTIRE2026Dataset(
#         root_dir=CONFIG['DATA_ROOT'], 
#         split='train', 
#         patch_size=CONFIG['PATCH_SIZE']
#     )
#     val_dataset = NTIRE2026Dataset(
#         root_dir=CONFIG['DATA_ROOT'], 
#         split='val', 
#         patch_size=CONFIG['PATCH_SIZE']
#     )

#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=CONFIG['BATCH_SIZE'], 
#         shuffle=True, 
#         num_workers=CONFIG['NUM_WORKERS'],
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=1, # Validate 1 image at a time for accurate metrics
#         shuffle=False, 
#         num_workers=CONFIG['NUM_WORKERS']
#     )

#     print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples")

#     # 2. Model, Optim, Loss
#     model = Restormer().to(CONFIG['DEVICE'])
#     criterion = CharbonnierLoss().to(CONFIG['DEVICE'])
#     optimizer = AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-4)
#     scaler = GradScaler()
#     scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)
    
#     # Metrics
#     scorer = NTIREMetric(device=CONFIG['DEVICE'])

#     best_score = -9999

#     # 3. Training Loop
#     for epoch in range(1, CONFIG['EPOCHS'] + 1):
#         model.train()
#         train_loss = 0
#         loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['EPOCHS']}")
        
#         for batch_idx, (inp, tar) in enumerate(loop):
#             inp, tar = inp.to(CONFIG['DEVICE']), tar.to(CONFIG['DEVICE'])
            
#             optimizer.zero_grad()
            
#             with autocast():  # <--- Runs math in float16 where safe
#                 pred = model(inp)
#                 loss = criterion(pred, tar)
            
#             scaler.scale(loss).backward() # <--- Scaled backward pass
#             scaler.step(optimizer)
#             scaler.update()
#             # -----------------------------
#             loop.set_postfix(loss=loss.item())
            
#         scheduler.step()

#         # 4. Validation Loop
#         if epoch % CONFIG['VAL_INTERVAL'] == 0:
#             model.eval()
#             val_score_accum = 0
#             val_psnr_accum = 0
            
#             with torch.no_grad():
#                 for inp, tar in tqdm(val_loader, desc="Validating"):
#                     inp, tar = inp.to(CONFIG['DEVICE']), tar.to(CONFIG['DEVICE'])
#                     pred = model(inp)
                    
#                     # Clamp to [0, 1] before metric calc
#                     pred = torch.clamp(pred, 0, 1)
                    
#                     metrics = scorer.calculate(pred, tar)
#                     val_score_accum += metrics['score']
#                     val_psnr_accum += metrics['psnr_y']
            
#             avg_score = val_score_accum / len(val_loader)
#             avg_psnr = val_psnr_accum / len(val_loader)
            
#             print(f"==> Validation Score: {avg_score:.4f} | PSNR_Y: {avg_psnr:.4f}")
            
#             # Save Best Model
#             if avg_score > best_score:
#                 best_score = avg_score
#                 torch.save(model.state_dict(), os.path.join(CONFIG['SAVE_DIR'], 'best_model.pth'))
#                 print("==> New Best Model Saved!")
                
#         # Save Regular Checkpoint
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#         }, os.path.join(CONFIG['SAVE_DIR'], 'latest.pth'))

# if __name__ == '__main__':
#     train()
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Import custom modules
from data.dataset import NTIRE2026Dataset
from models.restormer import Restormer
from models.losses import CharbonnierLoss, EdgeLoss
from utils.metrics import NTIREMetric

# --- Configuration ---
CONFIG = {
    'EPOCHS': 20,
    'BATCH_SIZE': 12,       
    'PATCH_SIZE': 128,     
    'LR': 1e-5,            
    'NUM_WORKERS': 4,      
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATA_ROOT': '/media/admin1/DL/ankit/raindrop/ntire2026-dualfocus-raindrop/data/train', 
    'SAVE_DIR': './checkpoints',
    'VAL_INTERVAL': 2     
}

def train():
    os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
    log_file = os.path.join(CONFIG['SAVE_DIR'], "training_log.txt")
    
    # 1. Dataset & Dataloaders
    train_dataset = NTIRE2026Dataset(
        root_dir=CONFIG['DATA_ROOT'], 
        split='train', 
        patch_size=CONFIG['PATCH_SIZE']
    )
    val_dataset = NTIRE2026Dataset(
        root_dir=CONFIG['DATA_ROOT'], 
        split='val', 
        patch_size=CONFIG['PATCH_SIZE']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=CONFIG['NUM_WORKERS']
    )

    # 2. Model, Optim, Loss
    model = Restormer().to(CONFIG['DEVICE'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    criterion = CharbonnierLoss().to(CONFIG['DEVICE'])
    criterion_edge = EdgeLoss().to(CONFIG['DEVICE'])
    optimizer = AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)
    scaler = GradScaler() 
    scorer = NTIREMetric(device=CONFIG['DEVICE'])
    
    
    # 3. Resume Logic (Fine-Tuning Setup)
    best_ckpt = os.path.join(CONFIG['SAVE_DIR'], 'best_model.pth')
    if os.path.exists(best_ckpt):
        print(f"Loading best weights for fine-tuning from {best_ckpt}...")
        model.load_state_dict(torch.load(best_ckpt, map_location=CONFIG['DEVICE'], weights_only=True))
    
    
    best_score = -9999
    start_epoch = 1

    # 3. Resume Logic
    latest_ckpt = os.path.join(CONFIG['SAVE_DIR'], 'latest.pth')
    # if os.path.exists(latest_ckpt):
    #     print(f"Resuming training from {latest_ckpt}...")
    #     checkpoint = torch.load(latest_ckpt, map_location=CONFIG['DEVICE'])
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scaler.load_state_dict(checkpoint['scaler_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     if 'best_score' in checkpoint:
    #         best_score = checkpoint['best_score']

    # 4. Training Loop
    for epoch in range(start_epoch, CONFIG['EPOCHS'] + 1):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['EPOCHS']}")
        
        for batch_idx, (inp, tar) in enumerate(loop):
            inp, tar = inp.to(CONFIG['DEVICE']), tar.to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            
            with autocast():
                pred = model(inp)
                # loss = criterion(pred, tar)
                # Calculate both losses
                loss_char = criterion(pred, tar)
                loss_edge = criterion_edge(pred, tar)
                
                # Combine them (0.05 is a standard empirical weight for edge loss)
                loss = loss_char + (0.05 * loss_edge)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Log Training Loss
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch} - Train Loss: {avg_train_loss:.6f}\n")

        # 5. Validation Loop
        if epoch % CONFIG['VAL_INTERVAL'] == 0:
            model.eval()
            val_score_accum = 0
            val_psnr_accum = 0
            
            with torch.no_grad():
                for inp, tar in tqdm(val_loader, desc="Validating"):
                    inp, tar = inp.to(CONFIG['DEVICE']), tar.to(CONFIG['DEVICE'])
                    pred = model(inp)
                    pred = torch.clamp(pred, 0, 1)
                    
                    metrics = scorer.calculate(pred, tar)
                    val_score_accum += metrics['score']
                    val_psnr_accum += metrics['psnr_y']
            
            avg_score = val_score_accum / len(val_loader)
            avg_psnr = val_psnr_accum / len(val_loader)
            
            print(f"==> Validation Score: {avg_score:.4f} | PSNR_Y: {avg_psnr:.4f}")
            
            # Log Validation Metrics
            with open(log_file, "a") as f:
                f.write(f"Epoch {epoch} - Val Score: {avg_score:.4f} | Val PSNR_Y: {avg_psnr:.4f}\n")
            
            if avg_score > best_score:
                best_score = avg_score
                torch.save(model.state_dict(), os.path.join(CONFIG['SAVE_DIR'], 'best_model.pth'))
                print("==> New Best Model Saved!")
                
        # Save Regular Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_score': best_score
        }, latest_ckpt)

if __name__ == '__main__':
    train()