import os
import random
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms  # <--- ADDED THIS IMPORT

class NTIRE2026Dataset(Dataset):
    def __init__(self, root_dir, split='train', split_ratio=0.9, patch_size=256):
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.samples = []

        # 1. Inspect Hierarchy
        # Use os.path.join for cross-platform compatibility
        day_path = os.path.join(root_dir, 'daytime', 'Clear', '*')
        night_path = os.path.join(root_dir, 'nighttime', 'Clear', '*')
        
        day_scenes = sorted(glob(day_path))
        night_scenes = sorted(glob(night_path))
        
        if not day_scenes and not night_scenes:
            # Fallback for checking if paths are correct
            print(f"Warning: No scenes found in {day_path} or {night_path}")
            
        all_scenes = day_scenes + night_scenes
        
        # 2. Split
        split_idx = int(len(all_scenes) * split_ratio)
        if split == 'train':
            selected_scenes = all_scenes[:split_idx]
        else:
            selected_scenes = all_scenes[split_idx:]

        # 3. Build Pairs
        for scene_path in selected_scenes:
            scene_id = os.path.basename(scene_path)
            path_parts = scene_path.split(os.sep)
            
            try:
                clear_idx = path_parts.index('Clear')
                time_of_day = path_parts[clear_idx - 1]
            except ValueError:
                continue 

            gt_images = sorted(glob(os.path.join(scene_path, '*.png')))
            
            for gt_path in gt_images:
                img_name = os.path.basename(gt_path)
                
                drop_path = os.path.join(root_dir, time_of_day, 'Drop', scene_id, img_name)
                blur_path = os.path.join(root_dir, time_of_day, 'Blur', scene_id, img_name)

                if os.path.exists(drop_path):
                    self.samples.append({'input': drop_path, 'target': gt_path, 'type': 'Drop'})
                
                if os.path.exists(blur_path):
                    self.samples.append({'input': blur_path, 'target': gt_path, 'type': 'Blur'})

        print(f"[{split.upper()}] Loaded {len(self.samples)} pairs from {len(selected_scenes)} scenes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pair = self.samples[idx]
        # Check if a sharp pseudo_gt exists for this specific scene
        target_path = pair['target']
        path_parts = target_path.split(os.sep)
        
        try:
            clear_idx = path_parts.index('clear')
            time_of_day = path_parts[clear_idx - 1]
            scene_id = path_parts[clear_idx + 1]
            
            pseudo_path = os.path.join(self.root_dir, time_of_day, 'pseudo_gt', f"pseudo_gt_{scene_id}.png")
            
            # If the script successfully made a sharp background for this folder, use it
            if os.path.exists(pseudo_path):
                target_path = pseudo_path
        except ValueError:
            pass
        
        try:
            inp = Image.open(pair['input']).convert('RGB')
            tar = Image.open(pair['target']).convert('RGB')
        except Exception as e:
            print(f"Error loading {pair['input']}: {e}")
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)

        # Apply Transforms
        if self.split == 'train':
            # --- THE FIX IS HERE ---
            # Use transforms.RandomCrop (Class) to get params, not TF (Functional)
            i, j, h, w = transforms.RandomCrop.get_params(inp, output_size=(self.patch_size, self.patch_size))
            
            inp = TF.crop(inp, i, j, h, w)
            tar = TF.crop(tar, i, j, h, w)

            # Horizontal Flip
            if random.random() > 0.5:
                inp = TF.hflip(inp)
                tar = TF.hflip(tar)
                
            # Vertical Flip
            if random.random() > 0.5:
                inp = TF.vflip(inp)
                tar = TF.vflip(tar)

        else:
            # Validation: Center Crop
            inp = TF.center_crop(inp, (self.patch_size, self.patch_size))
            tar = TF.center_crop(tar, (self.patch_size, self.patch_size))

        inp = TF.to_tensor(inp)
        tar = TF.to_tensor(tar)

        return inp, tar