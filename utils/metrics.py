import torch
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

class NTIREMetric:
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize LPIPS (AlexNet)
        # Using net='alex' as per standard restoration challenges
        self.lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    def rgb_to_y(self, image):
        """
        Converts RGB (B, 3, H, W) to Y (B, 1, H, W)
        """
        if image.shape[1] != 3:
            return image # Already 1 channel or invalid
            
        r, g, b = image.split(1, dim=1)
        # Standard BT.601 conversion
        y = 0.257 * r + 0.504 * g + 0.098 * b + (16 / 255.0)
        return y

    def calculate(self, preds, target):
        """
        preds: (B, 3, H, W) range [0, 1]
        target: (B, 3, H, W) range [0, 1]
        Returns: Dict with score and individual metrics
        """
        preds = preds.to(self.device)
        target = target.to(self.device)
        
        # 1. Y-Channel Metrics
        preds_y = self.rgb_to_y(preds)
        target_y = self.rgb_to_y(target)
        
        # PSNR (Y)
        val_psnr = psnr(preds_y, target_y, data_range=1.0)
        
        # SSIM (Y)
        val_ssim = ssim(preds_y, target_y, data_range=1.0)
        
        # 2. RGB LPIPS
        # LPIPS expects [-1, 1]
        preds_norm = (preds * 2) - 1
        target_norm = (target * 2) - 1
        val_lpips = self.lpips_fn(preds_norm, target_norm).mean()
        
        # 3. Final Formula
        score = val_psnr + (10 * val_ssim) - (5 * val_lpips)
        
        return {
            "score": score.item(),
            "psnr_y": val_psnr.item(),
            "ssim_y": val_ssim.item(),
            "lpips": val_lpips.item()
        }