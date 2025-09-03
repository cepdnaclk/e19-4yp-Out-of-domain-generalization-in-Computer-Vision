import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
from typing import Tuple, Optional, Union
from util import load_clip_model, DEVICE, CONTEXT_LENGTH, CONFIG_PATH, WEIGHTS_PATH, MODEL_NAME

class BiomedCLIPSaliencyGenerator:
    """
    Generate saliency maps for BiomedCLIP model to visualize regions of focus
    during classification tasks like tumor detection.
    """
    
    def __init__(self, config_path: str = None, weights_path: str = None, model_name: str = None):
        """
        Initialize the saliency generator with BiomedCLIP model.
        Uses default paths from util.py if not specified.
        
        Args:
            config_path: Path to model config JSON (uses CONFIG_PATH from util.py if None)
            weights_path: Path to model weights (uses WEIGHTS_PATH from util.py if None)
            model_name: Name of the model (uses MODEL_NAME from util.py if None)
        """
        self.device = DEVICE
        self.model, self.preprocess, self.tokenizer = load_clip_model(
            config_path=config_path or CONFIG_PATH,
            weights_path=weights_path or WEIGHTS_PATH,
            model_name=model_name or MODEL_NAME,
            device=self.device
        )
        self.model.eval()
        
        # Enable gradients for the visual encoder
        for param in self.model.visual.parameters():
            param.requires_grad_(True)
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Load and preprocess image for the model.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (preprocessed_tensor, original_pil_image)
        """
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        
        # Preprocess for model
        processed_image = self.preprocess(original_image).unsqueeze(0).to(self.device)
        processed_image.requires_grad_(True)
        
        return processed_image, original_image
    
    def get_text_features(self, positive_prompt: str, negative_prompt: str = None) -> torch.Tensor:
        """
        Encode text prompts to get text features.
        
        Args:
            positive_prompt: Prompt for positive class (e.g., "tumor present")
            negative_prompt: Prompt for negative class (e.g., "no tumor")
            
        Returns:
            Text features tensor
        """
        if negative_prompt is None:
            negative_prompt = "normal tissue"
        
        text_inputs = self.tokenizer(
            [negative_prompt, positive_prompt],
            context_length=CONTEXT_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        return text_features
    
    def generate_gradient_saliency(self, 
                                 image_path: str, 
                                 positive_prompt: str, 
                                 negative_prompt: str = None,
                                 target_class: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate gradient-based saliency map.
        
        Args:
            image_path: Path to input image
            positive_prompt: Positive class prompt
            negative_prompt: Negative class prompt
            target_class: Target class for gradient computation (0 or 1)
            
        Returns:
            Tuple of (saliency_map, original_image_array)
        """
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Get text features
        text_features = self.get_text_features(positive_prompt, negative_prompt)
        
        # Forward pass
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Compute logits
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.t())
        probs = F.softmax(logits, dim=1)
        
        # Compute gradients with respect to target class
        target_score = probs[0, target_class]
        target_score.backward()
        
        # Get gradients from input image
        gradients = image_tensor.grad.data
        
        # Compute saliency map (using absolute values of gradients)
        saliency = torch.abs(gradients).max(dim=1)[0].squeeze().cpu().numpy()
        
        # Convert original image to array
        original_array = np.array(original_image)
        
        return saliency, original_array
    
    def generate_gradcam(self,
                        image_path: str,
                        positive_prompt: str,
                        negative_prompt: str = None,
                        target_class: int = 1,
                        target_layer: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM visualization.
        
        Args:
            image_path: Path to input image
            positive_prompt: Positive class prompt  
            negative_prompt: Negative class prompt
            target_class: Target class for gradient computation
            target_layer: Name of target layer (if None, uses last conv layer)
            
        Returns:
            Tuple of (gradcam_heatmap, original_image_array)
        """
        # Variables to store gradients and activations
        gradients = None
        activations = None
        
        def save_gradient(grad):
            nonlocal gradients
            gradients = grad
        
        def save_activation(module, input, output):
            nonlocal activations
            activations = output
        
        # Find target layer (last convolutional layer by default)
        if target_layer is None:
            # For vision transformers, we'll use the last layer before classification
            # This might need adjustment based on BiomedCLIP architecture
            target_module = None
            for name, module in self.model.visual.named_modules():
                if 'blocks' in name and isinstance(module, torch.nn.Module):
                    target_module = module
            
            if target_module is None:
                # Fallback to a more general approach
                modules = list(self.model.visual.children())
                target_module = modules[-2] if len(modules) > 1 else modules[-1]
        else:
            target_module = dict(self.model.visual.named_modules())[target_layer]
        
        # Register hooks
        handle_gradient = target_module.register_backward_hook(save_gradient)
        handle_activation = target_module.register_forward_hook(save_activation)
        
        try:
            # Preprocess image
            image_tensor, original_image = self.preprocess_image(image_path)
            
            # Get text features
            text_features = self.get_text_features(positive_prompt, negative_prompt)
            
            # Forward pass
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Compute logits
            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.t())
            probs = F.softmax(logits, dim=1)
            
            # Backward pass
            target_score = probs[0, target_class]
            target_score.backward()
            
            # Generate Grad-CAM
            if gradients is not None and activations is not None:
                # Pool gradients across spatial dimensions
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                
                # Weight activations by gradients
                for i in range(activations.size()[1]):
                    activations[:, i, :, :] *= pooled_gradients[i]
                
                # Create heatmap
                heatmap = torch.mean(activations, dim=1).squeeze()
                heatmap = F.relu(heatmap)
                heatmap = F.interpolate(
                    heatmap.unsqueeze(0).unsqueeze(0),
                    size=(original_image.height, original_image.width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().cpu().numpy()
                
                # Normalize heatmap
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            else:
                print("Warning: Could not generate Grad-CAM. Using gradient saliency instead.")
                return self.generate_gradient_saliency(image_path, positive_prompt, negative_prompt, target_class)
        
        finally:
            # Remove hooks
            handle_gradient.remove()
            handle_activation.remove()
        
        # Convert original image to array
        original_array = np.array(original_image)
        
        return heatmap, original_array
    
    def save_saliency_map(self,
                         image_path: str,
                         positive_prompt: str,
                         negative_prompt: str = None,
                         output_path: str = None,
                         method: str = 'gradient',
                         target_class: int = 1,
                         alpha: float = 0.4,
                         colormap: str = 'jet') -> str:
        """
        Generate and save saliency map visualization.
        
        Args:
            image_path: Path to input image
            positive_prompt: Positive class prompt (e.g., "malignant tumor")
            negative_prompt: Negative class prompt (e.g., "benign tissue")
            output_path: Path to save the visualization (if None, auto-generated)
            method: Saliency method ('gradcam' or 'gradient')
            target_class: Target class (1 for positive, 0 for negative)
            alpha: Transparency for overlay (0.0 to 1.0)
            colormap: Matplotlib colormap for heatmap
            
        Returns:
            Path to saved visualization
        """
        # Generate saliency map
        if method.lower() == 'gradcam':
            try:
                saliency_map, original_image = self.generate_gradcam(
                    image_path, positive_prompt, negative_prompt, target_class
                )
            except Exception as e:
                print(f"Grad-CAM failed: {e}. Falling back to gradient saliency.")
                saliency_map, original_image = self.generate_gradient_saliency(
                    image_path, positive_prompt, negative_prompt, target_class
                )
                method = 'gradient'
        else:
            saliency_map, original_image = self.generate_gradient_saliency(
                image_path, positive_prompt, negative_prompt, target_class
            )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Saliency map
        im1 = axes[1].imshow(saliency_map, cmap=colormap)
        axes[1].set_title(f'Saliency Map ({method.title()})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(original_image)
        
        # Resize saliency map to match original image dimensions if needed
        if saliency_map.shape != original_image.shape[:2]:
            saliency_map_resized = cv2.resize(
                saliency_map, 
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            saliency_map_resized = saliency_map
        
        # Apply colormap and overlay
        colored_saliency = cm.get_cmap(colormap)(saliency_map_resized)
        axes[2].imshow(colored_saliency[:, :, :3], alpha=alpha)
        axes[2].set_title(f'Overlay (α={alpha})')
        axes[2].axis('off')
        
        # Add information about prompts and prediction
        class_name = "Positive" if target_class == 1 else "Negative"
        fig.suptitle(f'Saliency Analysis - Target: {class_name} Class\n'
                    f'Positive: "{positive_prompt}" | Negative: "{negative_prompt or "normal tissue"}"',
                    fontsize=12, y=0.95)
        
        plt.tight_layout()
        
        # Save visualization
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, f"{base_name}_saliency_{method}.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saliency map saved to: {output_path}")
        return output_path
    
    def analyze_image_focus(self,
                           image_path: str,
                           positive_prompt: str,
                           negative_prompt: str = None,
                           threshold: float = 0.7) -> dict:
        """
        Analyze where the model focuses and provide interpretation.
        
        Args:
            image_path: Path to input image
            positive_prompt: Positive class prompt
            negative_prompt: Negative class prompt
            threshold: Threshold for high attention regions
            
        Returns:
            Dictionary with analysis results
        """
        # Generate saliency map
        saliency_map, original_image = self.generate_gradcam(
            image_path, positive_prompt, negative_prompt, target_class=1
        )
        
        # Normalize saliency map
        saliency_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        # Find high attention regions
        high_attention_mask = saliency_normalized > threshold
        high_attention_percentage = np.sum(high_attention_mask) / high_attention_mask.size * 100
        
        # Calculate center of attention
        y_indices, x_indices = np.where(high_attention_mask)
        if len(y_indices) > 0:
            center_y = np.mean(y_indices) / saliency_map.shape[0]
            center_x = np.mean(x_indices) / saliency_map.shape[1]
        else:
            center_y, center_x = 0.5, 0.5
        
        # Calculate attention distribution
        attention_std = np.std(saliency_normalized)
        attention_mean = np.mean(saliency_normalized)
        
        return {
            'high_attention_percentage': high_attention_percentage,
            'attention_center': (center_x, center_y),
            'attention_concentration': attention_std,
            'average_attention': attention_mean,
            'max_attention': np.max(saliency_normalized),
            'saliency_map': saliency_normalized
        }


def main():
    """
    Example usage of the saliency generator.
    Uses default configuration from util.py
    """
    generator = BiomedCLIPSaliencyGenerator()
    
    # Example usage
    image_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/patches/patient_004_node_4/patch_patient_004_node_4_x_3328_y_21792.png"
    prompt_pair = ('Lymphocytes have smooth, round nuclei with condensed chromatin and normal CD5 expression.', 'Tumor cells display irregular nuclei with coarse chromatin and altered CD5 expression.')

    positive_prompt = prompt_pair[1]
    negative_prompt = prompt_pair[0]

    # Generate and save saliency map
    output_path = generator.save_saliency_map(
        image_path=image_path,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        method='gradient',
        target_class=1,
        alpha=0.4,
        output_path="./saliency_map.png"
    )
    
    # Analyze focus regions
    analysis = generator.analyze_image_focus(
        image_path=image_path,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt
    )
    
    print(f"Analysis Results:")
    print(f"High attention regions: {analysis['high_attention_percentage']:.2f}%")
    print(f"Attention center: ({analysis['attention_center'][0]:.3f}, {analysis['attention_center'][1]:.3f})")
    print(f"Attention concentration: {analysis['attention_concentration']:.3f}")


if __name__ == "__main__":
    main()
