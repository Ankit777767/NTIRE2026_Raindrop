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
    'INPUT_DIR': '/media/admin1/DL/ankit/raindrop/ntire2026-dualfocus-raindrop/data/codabench',
    'OUTPUT_DIR': './submission',
    'CHECKPOINT': './checkpoints/best_model.pth',
    # Changed to 'cuda' so it picks up the first available device visible to the process
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'FACTOR': 8
}

def check_image_size(x, factor):
    _, _, h, w = x.size()
    h_pad = (factor - (h % factor)) % factor
    w_pad = (factor - (w % factor)) % factor
    x_padded = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
    return x_padded, h, w

def run_inference():
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    
    # Verify which GPU is being used
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Loading model from {CONFIG['CHECKPOINT']}...")
    model = Restormer().to(CONFIG['DEVICE'])
    
    checkpoint = torch.load(CONFIG['CHECKPOINT'], map_location=CONFIG['DEVICE'])
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    image_paths = sorted(glob(os.path.join(CONFIG['INPUT_DIR'], '*.*')))
    print(f"Found {len(image_paths)} images.")
    
    if len(image_paths) == 0:
        return

    print("Starting inference...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            # 1. Clear cache before each image to help with OOM
            torch.cuda.empty_cache() 
            
            img_name = os.path.basename(img_path)
            inp = Image.open(img_path).convert('RGB')
            
            # 2. Convert to tensor and move to device
            inp_tensor = TF.to_tensor(inp).unsqueeze(0).to(CONFIG['DEVICE'])
            
            # Pad
            inp_padded, h_orig, w_orig = check_image_size(inp_tensor, CONFIG['FACTOR'])
            
            # 3. Forward Pass (use autocast if your GPU supports it to save memory)
            with torch.cuda.amp.autocast():
                restored_padded = model(inp_padded)
            
            # Unpad
            restored = restored_padded[:, :, :h_orig, :w_orig]
            restored = torch.clamp(restored, 0, 1)
            
            # Save
            save_path = os.path.join(CONFIG['OUTPUT_DIR'], img_name)
            TF.to_pil_image(restored.squeeze(0).cpu()).save(save_path)

            # Cleanup to free memory immediately
            del inp_tensor, inp_padded, restored_padded, restored

    print(f"\nDone! Results saved to {CONFIG['OUTPUT_DIR']}")

if __name__ == '__main__':
    run_inference()