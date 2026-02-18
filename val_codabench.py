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
    'INPUT_DIR': './Val_Input',       # Folder containing the 406 mixed images
    'OUTPUT_DIR': './submission',      # Where to save the restored images
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
    
    # Calculate how much to pad
    h_pad = (factor - (h % factor)) % factor
    w_pad = (factor - (w % factor)) % factor
    
    # Pad (left, right, top, bottom)
    # Reflect padding minimizes border artifacts
    x_padded = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
    
    return x_padded, h, w

def run_inference():
    # 1. Setup
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    print(f"Loading model from {CONFIG['CHECKPOINT']}...")
    model = Restormer().to(CONFIG['DEVICE'])
    
    # Load weights
    checkpoint = torch.load(CONFIG['CHECKPOINT'], map_location=CONFIG['DEVICE'])
    
    # Handle cases where checkpoint saves 'model_state_dict' or just the dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 2. Get Images
    # Supports png, jpg, jpeg
    image_paths = sorted(glob(os.path.join(CONFIG['INPUT_DIR'], '*.*')))
    print(f"Found {len(image_paths)} images in {CONFIG['INPUT_DIR']}")
    
    if len(image_paths) == 0:
        print("Error: No images found. Check your INPUT_DIR path.")
        return

    # 3. Inference Loop
    print("Starting inference...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            # Load Image
            img_name = os.path.basename(img_path)
            inp = Image.open(img_path).convert('RGB')
            inp_tensor = TF.to_tensor(inp).unsqueeze(0).to(CONFIG['DEVICE'])
            
            # Pad if necessary
            inp_padded, h_orig, w_orig = check_image_size(inp_tensor, CONFIG['FACTOR'])
            
            # Forward Pass
            restored_padded = model(inp_padded)
            
            # Unpad (Crop back to original size)
            restored = restored_padded[:, :, :h_orig, :w_orig]
            
            # Post-process
            restored = torch.clamp(restored, 0, 1)
            
            # Save
            save_path = os.path.join(CONFIG['OUTPUT_DIR'], img_name)
            TF.to_pil_image(restored.squeeze(0).cpu()).save(save_path)

    print(f"\nDone! Results saved to {CONFIG['OUTPUT_DIR']}")
    print("You can now zip the 'submission' folder and upload to Codabench.")

if __name__ == '__main__':
    run_inference()