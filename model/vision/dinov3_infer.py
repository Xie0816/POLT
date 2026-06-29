"""DINOv3 inference wrapper used by POLT perception."""

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from transformers import pipeline
from transformers.image_utils import load_image
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
from model.vision.vlad import VLAD, create_vlad_processor

class Dinov3Infer:
    """Load local DINOv3 weights and expose patch/VLAD feature inference."""

    def __init__(self, 
                 model_path = None, 
                 model_key = 'dinov3_vits16plus', 
                 pretrained_model_name = None,
                 use_vlad=False,
                 vlad_clusters=32,
                 vlad_cache_dir='model/vision/vlad_cache'
        ):
        
        # Resolve paths from the repository root so CLI and utility scripts agree.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        if model_path is None:
            model_path = os.path.join(project_root, 'model/third_party/dinov3')
        
        if pretrained_model_name is None:
            pretrained_model_name = os.path.join(project_root, 'weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
        
        print(f"DINOv3模型路径: {model_path}")
        print(f"预训练权重路径: {pretrained_model_name}")
        
        model = torch.hub.load(model_path, model_key, source='local', weights=pretrained_model_name)
        self.model = model.cuda()
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Optional VLAD-style clustering converts dense patch tokens to compact IDs.
        self.use_vlad = use_vlad
        self.vlad_clusters = vlad_clusters
        self.vlad_cache_dir = vlad_cache_dir
        self.vlad_processor = None
        
        if self.use_vlad:
            print(f"初始化VLAD处理器，聚类数: {vlad_clusters}")
            self.vlad_processor = create_vlad_processor(
                num_clusters=vlad_clusters,
                cache_dir=vlad_cache_dir
            )

    def model_infer(self, image, use_vlad = False):
        """Run DINOv3 inference and optionally map patch features through VLAD."""
        tensor_image = self.transform(image).unsqueeze(0)
        image_tensor = tensor_image.cuda()
        
        with torch.inference_mode():
            tokens, (H, W) = self.model.prepare_tokens_with_masks(image_tensor)

            print(f"Patch 分辨率: 高度 {H} patches, 宽度 {W} patches")
            print(f"总共 {H * W} 个 patches")
            
            features_dict = self.model.forward_features(image_tensor)
            patch_features = features_dict["x_norm_patchtokens"].squeeze(0)
        
        if use_vlad and self.vlad_processor is not None:
            vlad_features = self.vlad_processor.get_cluster_assignments(patch_features)
            
            print(f"原始特征形状: {patch_features.shape}")
            print(f"VLAD特征形状: {vlad_features.shape}")
            
            return vlad_features, patch_features, tokens, (H, W)
        else:
            return patch_features, tokens, (H, W)

    def compute_patch_similarity_heatmap(self, patch_features, H, W, target_patch_coord):
        """Compute a cosine-similarity heatmap from one target patch to all patches."""

        assert patch_features.shape[0] == H * W, f"特征数量{H*W}与网格大小{H}x{W}不匹配"
        
        target_idx = target_patch_coord[0] * W + target_patch_coord[1]
        target_feature = patch_features[target_idx]  # shape (feature_dim,)
        
        similarities = F.cosine_similarity(
            target_feature.unsqueeze(0),  # shape (1, feature_dim)
            patch_features,            # shape (num_patches, feature_dim)
            dim=1
        )
        
        heatmap = similarities.reshape(H, W).cpu().numpy()
        
        return heatmap

    def plot_similarity_heatmap(self, heatmap, target_patch_coord, original_image=None):
        """Plot a patch-similarity heatmap for offline feature inspection."""
        H, W = heatmap.shape
        
        if original_image is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            if isinstance(original_image, torch.Tensor):
                img_np = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # Normalize for display if the tensor was already standardized.
                if img_np.min() < 0:
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_np = np.array(original_image)
            
            ax1.imshow(img_np)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            patch_size = img_np.shape[0] // H
            target_h, target_w = target_patch_coord
            rect = patches.Rectangle(
                (target_w * patch_size, target_h * patch_size),
                patch_size, patch_size,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax1.add_patch(rect)
            
            im = ax2.imshow(heatmap, cmap='viridis', aspect='equal')
            
            ax2.plot(target_w, target_h, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            
            plt.colorbar(im, ax=ax2, label='Cosine Similarity')
            
            ax2.set_xlabel('Width (patch index)')
            ax2.set_ylabel('Height (patch index)')
            ax2.set_title(f'Cosine Similarity to Patch at ({target_h}, {target_w})')
            
            # Show patch-grid boundaries for visual inspection.
            ax2.set_xticks(np.arange(-0.5, W, 1), minor=True)
            ax2.set_yticks(np.arange(-0.5, H, 1), minor=True)
            ax2.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
            ax2.tick_params(which="minor", size=0)
            
            ax2.set_xticks(np.arange(0, W, max(1, W//10)))
            ax2.set_yticks(np.arange(0, H, max(1, H//10)))
            
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
            
            im = ax2.imshow(heatmap, cmap='viridis', aspect='equal')
            
            target_h, target_w = target_patch_coord
            ax2.plot(target_w, target_h, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            
            plt.colorbar(im, ax=ax2, label='Cosine Similarity')
            
            ax2.set_xlabel('Width (patch index)')
            ax2.set_ylabel('Height (patch index)')
            ax2.set_title(f'Cosine Similarity to Patch at ({target_h}, {target_w})')
            
            ax2.set_xticks(np.arange(-0.5, W, 1), minor=True)
            ax2.set_yticks(np.arange(-0.5, H, 1), minor=True)
            ax2.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
            ax2.tick_params(which="minor", size=0)
            
            ax2.set_xticks(np.arange(0, W, max(1, W//10)))
            ax2.set_yticks(np.arange(0, H, max(1, H//10)))
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax2 if original_image is None else (ax1, ax2)


if __name__ == "__main__":
    img_path = "/home/xie/Data/Terrain_Dataset/2025-02-25-14-40-20/image_front/1740465625684.png"
    image = Image.open(img_path)
    print(f"Image loaded, size: {image.size}")


    # Debug example 1: raw DINO patch features.
    # print("=" * 50)
    # print("Debug 1: DINO without VLAD")
    # print("=" * 50)
    # dino_sampler = Dinov3Infer()

    # patch_features, tokens, (H, W) = dino_sampler.model_infer(image)
    # print("patch_features.shape: ", patch_features.shape)
    
    # Plot a similarity heatmap from raw patch features.
    # print("\n" + "=" * 50)
    # print("Patch similarity heatmap")
    # print("=" * 50)
    # target_patch_coord = (60, 60)
    # heatmap = dino_sampler.compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord)
    
    # dino_sampler.plot_similarity_heatmap(heatmap, target_patch_coord, original_image=image)

    # Debug example 2: VLAD-compressed features.
    print("\n" + "=" * 50)
    print("Debug 2: DINO with VLAD")
    print("=" * 50)
    dino_sampler_vlad = Dinov3Infer(
        use_vlad=True
    )
    
    vlad_features, patch_features, tokens_vlad, (H, W) = dino_sampler_vlad.model_infer(
        image, 
        use_vlad=True
    )
    
    print(f"VLAD feature shape: {vlad_features.shape}")
    print(f"Raw patch feature shape: {patch_features.shape}")
    
    print("\n" + "=" * 50)
    print("Patch similarity heatmap")
    print("=" * 50)
    target_patch_coord = (60, 60)
    heatmap = dino_sampler_vlad.compute_patch_similarity_heatmap(vlad_features, H, W, target_patch_coord)
    
    dino_sampler_vlad.plot_similarity_heatmap(heatmap, target_patch_coord, original_image=image)
    
    print("\n" + "=" * 50)
    print("All debug checks completed.")
    print("=" * 50)
