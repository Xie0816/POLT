"""Train a VLAD vocabulary from an unlabeled folder of terrain images."""

import torch
import torch.nn.functional as F
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import cv2
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import setup_dino_project_paths

setup_dino_project_paths()
from model.vision.vlad import VLAD
from dinov3.hub.backbones import dinov3_vits16plus

class ImageFolderVLADTrainer:
    """Unsupervised VLAD trainer for one image folder."""
    
    def __init__(self, backbone_path, num_clusters=32, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_clusters = num_clusters
        
        print(f"Loading DINO backbone from: {backbone_path}")
        self.backbone = dinov3_vits16plus(
            pretrained=True,
            weights=backbone_path
        ).to(self.device)
        self.backbone.eval()

    def extract_patch_features(self, img_path, target_size=(600, 960)):
        """Extract DINO patch features from one image."""
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: return None
        
        # Match the input size used by the POLT DINO feature pipeline.
        img_resized = cv2.resize(img_bgr, (target_size[1], target_size[0]))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            features_dict = self.backbone.forward_features(img_tensor)
            patch_features = features_dict["x_norm_patchtokens"]
        return patch_features.squeeze(0).cpu()

    def train(self, input_folder, output_dir, max_images=1000, patches_per_img=1000):
        """Collect sampled patch descriptors, fit VLAD, and save the vocabulary."""
        if output_dir is None:
            output_dir = f'./outputs/vlad_models/vlad_custom_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(output_dir, exist_ok=True)

        # Scan supported image files and optionally subsample large folders.
        img_exts = ('.png', '.jpg', '.jpeg')
        img_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(img_exts)]
        if len(img_files) > max_images:
            img_files = np.random.choice(img_files, max_images, replace=False)

        all_feats = []
        print(f"正在从 {len(img_files)} 张图片提取特征...")
        for f in tqdm(img_files):
            feats = self.extract_patch_features(f)
            if feats is not None:
                # Limit descriptors per image to keep memory bounded.
                if len(feats) > patches_per_img:
                    idx = torch.randperm(len(feats))[:patches_per_img]
                    feats = feats[idx]
                all_feats.append(feats)
        
        train_data = torch.cat(all_feats, dim=0).to(self.device)
        print(f"特征收集完成: {train_data.shape}")

        # Fit VLAD centers from sampled patch descriptors.
        vlad_processor = VLAD(num_clusters=self.num_clusters, cache_dir=os.path.join(output_dir, 'vlad_cache'))
        vlad_processor.fit(train_data)

        # Keep the output format compatible with train_vlad_database.py.
        model_save_path = os.path.join(output_dir, 'vlad_model.pth')
        training_info = {
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_clusters': self.num_clusters,
            'feature_dim': train_data.shape[1],
            'num_training_samples': len(train_data),
            'source_folder': input_folder,
            'backbone': 'dinov3_vits16plus',
            'model_path': model_save_path
        }

        # Save the VLAD vocabulary as a PyTorch checkpoint.
        torch.save({
            'vlad_state': vlad_processor.__dict__,
            'training_info': training_info,
            'cluster_centers': vlad_processor.c_centers.cpu(),
            'num_clusters': self.num_clusters,
            'desc_dim': train_data.shape[1]
        }, model_save_path)

        # Save lightweight metadata for reproducibility.
        with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)

        print(f"训练完成！模型已保存至: {model_save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/xie/Data/Terrain_Dataset/2024-09-22-16-40-20/image_front')
    parser.add_argument('--backbone-path', type=str, default='weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth')
    parser.add_argument('--num-clusters', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='outputs/vlad_models/baotou')
    args = parser.parse_args()

    trainer = ImageFolderVLADTrainer(args.backbone_path, args.num_clusters)
    trainer.train(args.input, args.output_dir)

if __name__ == "__main__":
    main()
