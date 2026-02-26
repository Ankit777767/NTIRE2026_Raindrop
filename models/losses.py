import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """
    Robust L1 loss.
    L_char = sqrt( (X - Y)^2 + eps^2 )
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.eps * self.eps))
        return torch.mean(loss)

class EdgeLoss(nn.Module):
    """
    Helps recover the sharp edges of the background.
    Optimized for Multi-GPU environments.
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        # Create a 2D Gaussian kernel
        kernel = torch.matmul(k.t(), k) 
        # Reshape to (Out_Channels, In_Channels/Groups, H, W) -> (3, 1, 5, 5)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        
        # Register as a buffer so it automatically moves across GPUs in DataParallel
        self.register_buffer('kernel', kernel)
        
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = torch.nn.functional.pad(img, (kw//2, kw//2, kh//2, kh//2), mode='replicate')
        # Apply depthwise convolution using the registered buffer
        return torch.nn.functional.conv2d(img, self.kernel, groups=n_channels)

    def forward(self, x, y):
        laplacian_x = x - self.conv_gauss(x)
        laplacian_y = y - self.conv_gauss(y)
        return self.loss(laplacian_x, laplacian_y)