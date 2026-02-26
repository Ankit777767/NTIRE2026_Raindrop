import os
import torch
import torch.nn.functional as F
from glob import glob
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

# Import your model
from models.restormer import Restormer

# --- Configuration ---
CONFIG = {
    'INPUT_DIR': '/media/admin1/DL/ankit/raindrop/ntire2026-dualfocus-raindrop/data/codabench',       # Folder containing the 406 mixed images
    'OUTPUT_DIR': './submission',     # Where to save the restored images
    'CHECKPOINT': './checkpoints/best_model.pth', # Path to your best .pth file
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'FACTOR': 8  # Restormer requires dimensions to be multiples of 8
}

def check_image_size(x, factor):
    """
    Pads the image so height and width are multiples of 'factor'.
    Returns padded image and the original dimensions (h, w).
    """
    _, _, h, w = x.size()
    h_pad = (factor - (h % factor)) % factor
    w_pad = (factor - (w % factor)) % factor
    x_padded = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
    return x_padded, h, w

def forward_tta(model, x):
    """
    Applies 8x geometric Test-Time Augmentation.
    """
    # 1. Generate 8 augmented inputs
    x0 = x
    x1 = torch.rot90(x, 1, [2, 3])
    x2 = torch.rot90(x, 2, [2, 3])
    x3 = torch.rot90(x, 3, [2, 3])
    
    x4 = torch.flip(x, [3]) # Horizontal flip
    x5 = torch.rot90(x4, 1, [2, 3])
    x6 = torch.rot90(x4, 2, [2, 3])
    x7 = torch.rot90(x4, 3, [2, 3])
    
    aug_inputs = [x0, x1, x2, x3, x4, x5, x6, x7]
    aug_outputs = []
    
    # 2. Pass each through the model (iterative to save VRAM)
    for inp in aug_inputs:
        aug_outputs.append(model(inp))
        
    # 3. De-augment (reverse the transformations)
    y0 = aug_outputs[0]
    y1 = torch.rot90(aug_outputs[1], 3, [2, 3]) # Reverse of rot90(1) is rot90(3)
    y2 = torch.rot90(aug_outputs[2], 2, [2, 3])
    y3 = torch.rot90(aug_outputs[3], 1, [2, 3])
    
    y4 = torch.flip(aug_outputs[4], [3])
    y5 = torch.flip(torch.rot90(aug_outputs[5], 3, [2, 3]), [3])
    y6 = torch.flip(torch.rot90(aug_outputs[6], 2, [2, 3]), [3])
    y7 = torch.flip(torch.rot90(aug_outputs[7], 1, [2, 3]), [3])
    
    # 4. Average the predictions
    y_final = (y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7) / 8.0
    return y_final

def run_inference():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    print(f"Loading model from {CONFIG['CHECKPOINT']}...")
    model = Restormer().to(CONFIG['DEVICE'])
    
    # Wrap in DataParallel if you are using your 2 GPUs for inference too
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    checkpoint = torch.load(CONFIG['CHECKPOINT'], map_location=CONFIG['DEVICE'], weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    image_paths = sorted(glob(os.path.join(CONFIG['INPUT_DIR'], '*.*')))
    print(f"Found {len(image_paths)} images in {CONFIG['INPUT_DIR']}")
    
    if len(image_paths) == 0:
        print("Error: No images found. Check your INPUT_DIR path.")
        return

    print("Starting inference with 8x TTA...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            img_name = os.path.basename(img_path)
            inp = Image.open(img_path).convert('RGB')
            inp_tensor = TF.to_tensor(inp).unsqueeze(0).to(CONFIG['DEVICE'])
            
            # Pad if necessary
            inp_padded, h_orig, w_orig = check_image_size(inp_tensor, CONFIG['FACTOR'])
            
            # --- USE TTA FORWARD PASS ---
            with torch.amp.autocast('cuda'):
                restored_padded = forward_tta(model, inp_padded)
            
            # Unpad (Crop back to original size)
            restored = restored_padded[:, :, :h_orig, :w_orig]
            
            # Post-process
            restored = torch.clamp(restored, 0, 1)
            
            # Save
            save_path = os.path.join(CONFIG['OUTPUT_DIR'], img_name)
            TF.to_pil_image(restored.squeeze(0).cpu()).save(save_path)

    print(f"\nDone! Results saved to {CONFIG['OUTPUT_DIR']}")

if __name__ == '__main__':
    run_inference()