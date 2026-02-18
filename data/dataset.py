import os
import random
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torch

class NTIRE2026Dataset(Dataset):
    def __init__(self, root_dir, split='train', split_ratio=0.9, patch_size=256):
        """
        root_dir: Path to dataset root.
        split: 'train' or 'val'
        split_ratio: Percentage of scenes to keep for training.
        """
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.samples = []

        # 1. Inspect Hierarchy to find all scene folders
        # We look into 'clear' folders to find valid scene IDs.
        # Structure: root_dir/Daytime/clear/00001
        
        # Use os.path.join for cross-platform compatibility
        day_path = os.path.join(root_dir, 'Daytime', 'clear', '*')
        night_path = os.path.join(root_dir, 'Nighttime', 'clear', '*')
        
        day_scenes = sorted(glob(day_path))
        night_scenes = sorted(glob(night_path))
        
        if not day_scenes and not night_scenes:
            raise ValueError(f"No scenes found in {root_dir}. Check your path structure.")

        all_scenes = day_scenes + night_scenes
        
        # 2. Create Train/Val Split based on Scene IDs
        # We split by SCENE, not by image, to prevent data leakage.
        split_idx = int(len(all_scenes) * split_ratio)
        
        if split == 'train':
            selected_scenes = all_scenes[:split_idx]
        else:
            selected_scenes = all_scenes[split_idx:]

        # 3. Build the Pair List
        for scene_path in selected_scenes:
            # scene_path example: ...\Daytime\clear\00001 (Windows) or .../Daytime/clear/00001 (Linux)
            scene_id = os.path.basename(scene_path)
            
            # Extract 'Daytime' or 'Nighttime' safely
            # We go up 2 levels from the scene folder to find the time of day
            path_parts = scene_path.split(os.sep)
            # Find index of 'clear' and go one back
            try:
                clear_idx = path_parts.index('clear')
                time_of_day = path_parts[clear_idx - 1]
            except ValueError:
                continue # Skip if structure is unexpected

            gt_images = sorted(glob(os.path.join(scene_path, '*.png')))
            
            for gt_path in gt_images:
                img_name = os.path.basename(gt_path)
                
                # Input A: Drop focused (Background blur)
                drop_path = os.path.join(root_dir, time_of_day, 'drop', scene_id, img_name)
                
                # Input B: Blur focused (Drops blur)
                blur_path = os.path.join(root_dir, time_of_day, 'blur', scene_id, img_name)

                if os.path.exists(drop_path):
                    self.samples.append({'input': drop_path, 'target': gt_path, 'type': 'drop'})
                
                if os.path.exists(blur_path):
                    self.samples.append({'input': blur_path, 'target': gt_path, 'type': 'blur'})

        print(f"[{split.upper()}] Loaded {len(self.samples)} pairs from {len(selected_scenes)} scenes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pair = self.samples[idx]
        
        try:
            inp = Image.open(pair['input']).convert('RGB')
            tar = Image.open(pair['target']).convert('RGB')
        except Exception as e:
            print(f"Error loading {pair['input']}: {e}")
            # Return a dummy tensor or handle error appropriately
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)

        # Apply Transforms
        if self.split == 'train':
            # Random Crop
            i, j, h, w = TF.RandomCrop.get_params(inp, output_size=(self.patch_size, self.patch_size))
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
            # For validation, we might want center crop or full image
            # Here we do CenterCrop to ensure consistent size evaluation
            inp = TF.center_crop(inp, (self.patch_size, self.patch_size))
            tar = TF.center_crop(tar, (self.patch_size, self.patch_size))

        inp = TF.to_tensor(inp)
        tar = TF.to_tensor(tar)

        return inp, tar