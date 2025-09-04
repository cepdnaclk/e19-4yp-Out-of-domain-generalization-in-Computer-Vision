import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as filters
from PIL import Image
import os
import sys
from util import load_clip_model, DEVICE, CONTEXT_LENGTH, CONFIG_PATH, WEIGHTS_PATH, MODEL_NAME



def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map

def viz_attn(img, attn_map, blur=True, save_path=None):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[1].imshow(getAttMap(img, attn_map, blur))
    axes[1].set_title('GradCAM Heatmap')
    for ax in axes:
        ax.axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"GradCAM visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()  # Close the figure to free memory

def save_heatmap_overlay(img, attn_map, blur=True, save_path=None):
    """Save just the heatmap overlay without side-by-side comparison"""
    heatmap_img = getAttMap(img, attn_map, blur)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap_img)
    plt.axis("off")
    
    if save_path:
        # Create overlay filename
        base_name = os.path.splitext(save_path)[0]
        overlay_path = f"{base_name}_overlay.png"
        plt.savefig(overlay_path, bbox_inches='tight', dpi=300, pad_inches=0)
        print(f"Heatmap overlay saved to: {overlay_path}")
    
    plt.close()
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.



class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
        
        print(f"Gradient shape: {grad.shape}")
        print(f"Activation shape: {act.shape}")
    
        # Handle different tensor shapes for Vision Transformer vs CNN layers
        if len(grad.shape) == 3:  # Vision Transformer: (batch, seq_len, dim)
            print("Processing Vision Transformer output")
            # For ViT, we need to reshape to spatial dimensions
            # Assuming square patches, calculate patch grid size
            batch_size, seq_len, dim = grad.shape
            print(f"Original ViT shape: batch={batch_size}, seq_len={seq_len}, dim={dim}")
            
            # Skip CLS token if present (seq_len = num_patches + 1)
            if seq_len == 197:  # 14x14 patches + 1 CLS token for 224x224 input
                patch_size = 14
                grad = grad[:, 1:, :]  # Remove CLS token
                act = act[:, 1:, :]    # Remove CLS token
                seq_len = 196
                print("Removed CLS token, new seq_len:", seq_len)
            elif seq_len == 196:  # 14x14 patches for 224x224 input
                patch_size = 14
            else:
                # Calculate patch size dynamically
                patch_size = int(seq_len ** 0.5)
                print(f"Calculated patch_size: {patch_size}")
            
            # Reshape to spatial format: (batch, patch_h, patch_w, dim)
            grad = grad.reshape(batch_size, patch_size, patch_size, dim)
            act = act.reshape(batch_size, patch_size, patch_size, dim)
            
            # Transpose to (batch, dim, patch_h, patch_w) for compatibility
            grad = grad.permute(0, 3, 1, 2)
            act = act.permute(0, 3, 1, 2)
            print(f"Reshaped gradient: {grad.shape}")
            print(f"Reshaped activation: {act.shape}")
        
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        if len(grad.shape) == 4:  # (batch, channel/dim, height, width)
            alpha = grad.mean(dim=(2, 3), keepdim=True)
            print(f"Alpha shape: {alpha.shape}")
        else:  # Fallback for other shapes
            print(f"Using fallback for grad shape: {grad.shape}")
            alpha = grad.mean(dim=tuple(range(2, len(grad.shape))), keepdim=True)
            
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam


def generate_gradcam_for_biomedclip(image_path, caption, output_dir=".", saliency_layer="blocks", blur=True):
    """
    Simple function to generate GradCAM for BioMedCLIP given an image and caption.
    
    Args:
        image_path (str): Path to the input image
        caption (str): Text caption describing the image
        output_dir (str): Directory to save output images
        saliency_layer (str): Layer to visualize ("blocks", "norm_pre", "norm", "head")
        blur (bool): Whether to apply Gaussian blur to the heatmap
    
    Returns:
        numpy.ndarray: The attention map
    """
    device = DEVICE
    
    # Load BioMedCLIP model using checkpoints
    model, preprocess, tokenizer = load_clip_model()
    model = model.to(device).eval()
    print(f"Image path : {image_path}")
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Prepare inputs
    image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_np = load_image(image_path, 224)
    text_input = tokenizer([caption], context_length=CONTEXT_LENGTH).to(device)
    
    # Get target layer
    print(f"Model visual structure: {type(model.visual)}")
    if hasattr(model.visual, 'trunk'):
        print("Model has trunk attribute")
        if saliency_layer == "blocks":
            if hasattr(model.visual.trunk, 'blocks') and len(model.visual.trunk.blocks) > 0:
                target_layer = model.visual.trunk.blocks[-1]
                print(f"Using trunk.blocks[-1]: {type(target_layer)}")
            else:
                raise ValueError("No blocks found in model.visual.trunk")
        elif saliency_layer == "norm_pre":
            target_layer = model.visual.trunk.norm_pre
        elif saliency_layer == "norm":
            target_layer = model.visual.trunk.norm
        elif saliency_layer == "head":
            target_layer = model.visual.head if hasattr(model.visual, 'head') else model.visual.trunk.head
        else:
            target_layer = model.visual.trunk.blocks[-1]
    else:
        print("Model does not have trunk attribute")
        if saliency_layer == "blocks":
            if hasattr(model.visual, 'blocks') and len(model.visual.blocks) > 0:
                target_layer = model.visual.blocks[-1]
                print(f"Using blocks[-1]: {type(target_layer)}")
            else:
                # Try to find transformer blocks in other locations
                for attr_name in dir(model.visual):
                    attr = getattr(model.visual, attr_name)
                    if hasattr(attr, 'blocks') and len(attr.blocks) > 0:
                        target_layer = attr.blocks[-1]
                        print(f"Found blocks in {attr_name}: {type(target_layer)}")
                        break
                else:
                    raise ValueError("No blocks found in model.visual")
        elif saliency_layer == "norm_pre":
            target_layer = model.visual.norm_pre
        elif saliency_layer == "norm":
            target_layer = model.visual.norm
        elif saliency_layer == "head":
            target_layer = model.visual.head
        else:
            target_layer = model.visual.blocks[-1]
    
    print(f"Selected target layer: {target_layer}")
    print(f"Target layer type: {type(target_layer)}")
    
    # Generate GradCAM
    attn_map = gradCAM(
        model.visual,
        image_input,
        model.encode_text(text_input).float(),
        target_layer
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()
    
    # Create output paths
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    comparison_path = os.path.join(output_dir, f"{base_name}_gradcam_comparison.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_gradcam_overlay.png")
    
    # Save visualizations
    viz_attn(image_np, attn_map, blur, save_path=comparison_path)
    save_heatmap_overlay(image_np, attn_map, blur, save_path=overlay_path)
    
    return attn_map


