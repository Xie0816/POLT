"""Train a VLAD vocabulary from the RELLIS-3D semantic dataset."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import setup_dino_project_paths

setup_dino_project_paths()
from model.vision.vlad import VLAD
from utils.rellis3d_data.rellis3d import get_rellis3d_ontology_dataloaders, get_rellis3d_ontology_class_info
from dinov3.hub.backbones import dinov3_vits16plus


class DatabaseVLADTrainer:
    """Dataset-based VLAD trainer with class-balanced feature sampling."""
    
    def __init__(self, backbone_path, num_clusters=64, device='cuda', ontology_path=None):
        self.device = device
        self.num_clusters = num_clusters
        
        print(f"Loading DINO backbone from: {backbone_path}")
        self.backbone = dinov3_vits16plus(
            pretrained=True,
            weights=backbone_path
        ).to(device)
        
        self.backbone.eval()
        
        # Dataset ontology is needed for class-balanced descriptor sampling.
        if ontology_path is None:
            ontology_path = "/home/xie/Data/Rellis-3D/Rellis_3D_ontology/ontology.yaml"
        
        self.num_classes, self.class_names, self.class_colors, _, _ = get_rellis3d_ontology_class_info(ontology_path)
        
        print(f"VLAD训练器初始化完成，类别数: {self.num_classes}, 聚类数: {num_clusters}")
    
    def extract_features(self, images):
        """Extract DINO patch features for a batch of images."""
        with torch.no_grad():
            features_dict = self.backbone.forward_features(images)
            patch_features = features_dict["x_norm_patchtokens"]  # [B, N, C]
        return patch_features
    
    def collect_features_from_dataset(self, data_loader, max_samples_per_class=5000):
        """Collect class-balanced descriptors from image/mask batches."""
        class_feature_buffer = {class_id: [] for class_id in self.class_names.keys()}
        
        # TODO: adjust these IDs if a different ontology.yaml is used.
        CRITICAL_CLASSES = [13, 14, 15] 

        print(f"开始平衡采样特征，关键障碍物类别ID: {CRITICAL_CLASSES}")
        pbar = tqdm(data_loader, desc='Collecting features (Balanced)')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Align semantic masks with DINO patch descriptors.
            
            # 1. Extract patch descriptors.
            with torch.no_grad():
                # Call the backbone directly to avoid extra wrapper overhead.
                features_dict = self.backbone.forward_features(images)
                patch_features = features_dict["x_norm_patchtokens"] # [B, N, C]
                B, N, C = patch_features.shape
            
            # 2. Compute feature-map size assuming ViT-S/16 patching.
            #    images: [B, 3, H_img, W_img]
            H_img = images.shape[2]
            W_img = images.shape[3]
            H_feat = H_img // 16  # DINOv3 ViT-S/16 patch size.
            W_feat = W_img // 16
            
            # Validate the descriptor count against the expected patch grid.
            expected_N = H_feat * W_feat
            if N != expected_N:
                print(f"警告: 特征数N({N})与计算得到的特征图尺寸({H_feat}x{W_feat}={expected_N})不匹配，使用计算得到的尺寸")
                # Reshape descriptors if the computed grid is the intended layout.
                if N == expected_N:
                    # Shape already matches; keep the tensor as-is.
                    pass
                else:
                    # DINOv3 descriptors are normally row-major patch tokens.
                    patch_features = patch_features.reshape(B, H_feat, W_feat, C)
                    # Flatten back to [B, N, C].
                    patch_features = patch_features.reshape(B, expected_N, C)
                    N = expected_N
            
            # 3. Resize masks to match patch resolution.
            if masks.dim() == 3:
                masks = masks.unsqueeze(1) # (B, 1, H, W)
            
            # Nearest-neighbor interpolation preserves discrete label IDs.
            resized_masks = F.interpolate(
                masks.float(), 
                size=(H_feat, W_feat), 
                mode='nearest'
            ).long().squeeze(1) # (B, H_feat, W_feat)
            
            # 4. Flatten masks and descriptors for class-wise indexing.
            masks_flat = resized_masks.flatten(1) # (B, N), where N = H_feat * W_feat.
            patch_features_flat = patch_features.flatten(0, 1) # (B*N, C)
            masks_flat = masks_flat.flatten() # (B*N)
            
            # Collect features by class for the current batch.
            unique_classes_in_batch = torch.unique(masks_flat)
            
            for class_id in unique_classes_in_batch:
                class_id = class_id.item()
                
                # Skip classes outside the ontology collection list, such as void.
                if class_id not in class_feature_buffer:
                    continue
                
                # Downsample common classes while retaining critical obstacle classes.
                current_count = sum(len(c) for c in class_feature_buffer[class_id])
                if class_id not in CRITICAL_CLASSES and current_count > max_samples_per_class:
                    continue

                # Select descriptors for this class.
                class_mask = (masks_flat == class_id)
                selected_features = patch_features_flat[class_mask] # [M, C]
                
                if selected_features.shape[0] > 0:
                    # Store on CPU to prevent GPU memory growth.
                    class_feature_buffer[class_id].append(selected_features.cpu())
            
            # Drop large temporary tensors before the next batch.
            del patch_features, resized_masks, masks_flat, patch_features_flat
        
        # Balance and merge class-specific buffers.
        final_train_features = []
        print("\n=== Class Distribution Before Balancing ===")
        
        for class_id, feats_list in class_feature_buffer.items():
            if not feats_list:
                continue
                
            # Merge descriptor chunks for this class.
            all_feats_cat = torch.cat(feats_list, dim=0)
            count = len(all_feats_cat)
            
            class_name = self.class_names.get(class_id, {}).get('name', f'Class_{class_id}')
            print(f"Class {class_id} ({class_name}): Collected {count} samples")
            
            # Retain obstacle descriptors and downsample common terrain classes.
            if class_id not in CRITICAL_CLASSES and count > max_samples_per_class:
                # Randomly truncate over-represented classes.
                indices = torch.randperm(count)[:max_samples_per_class]
                selected = all_feats_cat[indices]
                print(f"  -> Downsampled to {max_samples_per_class}")
            else:
                # Keep all descriptors for critical or small classes.
                selected = all_feats_cat
                print(f"  -> Kept all")
            
            final_train_features.append(selected)
        
        # Merge all class-balanced descriptors into the final training set.
        if not final_train_features:
            print("警告：未收集到任何特征！请检查 Mask 值和 class_names 是否对应。")
            return torch.empty(0, 384)
            
        all_features_tensor = torch.cat(final_train_features, dim=0)
        
        # Shuffle to avoid class-ordered KMeans initialization artifacts.
        shuffled_indices = torch.randperm(len(all_features_tensor))
        all_features_tensor = all_features_tensor[shuffled_indices]
        
        print(f"\nTotal balanced features for VLAD training: {len(all_features_tensor)}")
        return all_features_tensor
       
    def train_vlad_from_database(self, data_loader, output_dir=None, 
                                max_train_samples=50000, 
                                sampling_strategy='uniform',
                                cache_dir=None):
        """
        Train a VLAD vocabulary from dataset descriptors.

        Args:
            data_loader: Dataset loader that yields images and masks.
            output_dir: Directory for checkpoints and metadata.
            max_train_samples: Maximum number of descriptors used for K-means.
            sampling_strategy: Final sampling policy when descriptor count is high.
            cache_dir: Optional VLAD cache directory.
        """
        if output_dir is None:
            output_dir = f'./outputs/vlad_models/vlad_database_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure VLAD cache directory.
        if cache_dir is None:
            cache_dir = os.path.join(output_dir, 'vlad_cache')
        
        print("开始从数据库收集特征...")
        
        # Estimate per-batch sampling budget when needed.
        total_batches = len(data_loader)
        max_samples_per_batch = max_train_samples // total_batches if max_train_samples < total_batches * 1000 else None
        
        # Collect descriptors from the dataset.
        train_features = self.collect_features_from_dataset(
            data_loader, 
            # max_samples_per_batch=max_samples_per_batch,
            # sampling_strategy=sampling_strategy
        )
        
        # Apply a final sampling pass if too many descriptors were collected.
        if len(train_features) > max_train_samples:
            print(f"收集了 {len(train_features)} 个特征，采样到 {max_train_samples} 个进行VLAD训练")
            if sampling_strategy == 'uniform':
                indices = torch.linspace(0, len(train_features) - 1, max_train_samples, dtype=torch.long)
            else:
                indices = torch.randperm(len(train_features))[:max_train_samples]
            train_features = train_features[indices]
        
        print(f"使用 {len(train_features)} 个特征训练VLAD，聚类数: {self.num_clusters}")
        
        # Create the VLAD processor.
        vlad_processor = VLAD(
            num_clusters=self.num_clusters,
            cache_dir=cache_dir
        )
        
        # Fit the VLAD vocabulary.
        print("开始训练VLAD...")
        train_features = train_features.to(self.device)
        vlad_processor.fit(train_features)
        
        # Save training metadata together with the vocabulary.
        training_info = {
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_clusters': self.num_clusters,
            'feature_dim': train_features.shape[1],
            'num_training_samples': len(train_features),
            'max_train_samples': max_train_samples,
            'sampling_strategy': sampling_strategy,
            'backbone': 'dinov3_vits16plus',
            'dataset': 'RELLIS-3D',
            'cache_dir': cache_dir,
            'model_path': os.path.join(output_dir, 'vlad_model.pth')
        }
        
        # Save the VLAD model checkpoint.
        model_save_path = os.path.join(output_dir, 'vlad_model.pth')
        torch.save({
            'vlad_state': vlad_processor.__dict__,
            'training_info': training_info,
            'cluster_centers': vlad_processor.c_centers,
            'num_clusters': self.num_clusters,
            'desc_dim': vlad_processor.desc_dim
        }, model_save_path)
        
        # Save metadata as JSON for easy inspection.
        info_save_path = os.path.join(output_dir, 'training_info.json')
        with open(info_save_path, 'w') as f:
            json.dump(training_info, f, indent=2, cls=NumpyEncoder)
        
        print(f"VLAD模型保存到: {model_save_path}")
        print(f"训练信息保存到: {info_save_path}")
        
        return vlad_processor
    
    def test_vlad_performance(self, vlad_processor, test_loader, num_test_samples=100):
        """
        Test VLAD compression behavior on a small number of samples.

        Args:
            vlad_processor: Fitted VLAD processor.
            test_loader: DataLoader used for evaluation samples.
            num_test_samples: Number of images to inspect.
        """
        print(f"测试VLAD性能，测试样本数: {num_test_samples}")
        
        test_count = 0
        compression_ratios = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(test_loader):
                if test_count >= num_test_samples:
                    break
                
                images = images.to(self.device)
                
                # Extract raw patch descriptors.
                patch_features = self.extract_features(images)  # [B, N, C]
                B, N, C = patch_features.shape
                
                for i in range(B):
                    if test_count >= num_test_samples:
                        break
                    
                    single_patch_features = patch_features[i]  # [N, C]
                    
                    # Generate VLAD features for one image.
                    vlad_features = vlad_processor.transform(
                        single_patch_features, 
                        reduce=True, 
                        mode='vlad'
                    )
                    
                    # Compute descriptor compression ratio.
                    original_size = N * C
                    vlad_size = len(vlad_features) if len(vlad_features.shape) == 1 else vlad_features.numel()
                    compression_ratio = original_size / vlad_size
                    compression_ratios.append(compression_ratio)
                    
                    test_count += 1
                    
                    if test_count % 10 == 0:
                        print(f"测试样本 {test_count}/{num_test_samples}, "
                              f"原始大小: {original_size}, VLAD大小: {vlad_size}, "
                              f"压缩比: {compression_ratio:.2f}")
        
        avg_compression = np.mean(compression_ratios)
        print(f"\n平均压缩比: {avg_compression:.2f}")
        print(f"压缩比范围: {np.min(compression_ratios):.2f} - {np.max(compression_ratios):.2f}")
        
        return {
            'avg_compression_ratio': avg_compression,
            'compression_ratios': compression_ratios
        }


def load_trained_vlad(model_path, device='cuda'):
    """
    Load a trained VLAD checkpoint.

    Args:
        model_path: Path to the saved checkpoint.
        device: Target device for cluster centers.
    """
    print(f"Loading VLAD model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create a VLAD processor from checkpoint metadata.
    vlad_processor = VLAD(
        num_clusters=checkpoint['num_clusters'],
        desc_dim=checkpoint['desc_dim']
    )
    
    # Rebuild nested VLAD state for compatibility with older checkpoints.
    vlad_processor.vlad = VLAD(
        num_clusters=checkpoint['num_clusters'],
        desc_dim=checkpoint['desc_dim']
    )
    
    # Restore cluster centers.
    vlad_processor.vlad.c_centers = checkpoint['cluster_centers'].to(device)
    vlad_processor.vlad.kmeans = None  # Rebuilt lazily if needed.
    vlad_processor.is_fitted = True
    
    print(f"VLAD模型加载完成，聚类数: {checkpoint['num_clusters']}, 特征维度: {checkpoint['desc_dim']}")
    
    return vlad_processor, checkpoint['training_info']


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that serializes numpy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description='从数据库训练VLAD模型')
    parser.add_argument('--backbone-path', type=str, 
                        default='weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth', 
                        help='DINO backbone权重路径')
    parser.add_argument('--num-clusters', type=int, default=16, help='VLAD聚类数量')
    parser.add_argument('--batch-size', type=int, default=16, help='批处理大小')
    parser.add_argument('--image-size', type=int, nargs=2, default=[600, 960], help='图像大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--num-workers', type=int, default=8, help='数据加载器工作线程数')
    parser.add_argument('--data-root', type=str, default="/home/xie/Data/Rellis-3D", 
                        help='RELLIS-3D数据根目录')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                        default='train', help='数据集分割')
    parser.add_argument('--max-train-samples', type=int, default=50000, 
                        help='最大训练样本数')
    parser.add_argument('--sampling-strategy', type=str, choices=['uniform', 'random'], 
                        default='uniform', help='采样策略')
    parser.add_argument('--cache-dir', type=str, default=None, help='VLAD缓存目录')
    parser.add_argument('--test-performance', action='store_true', 
                        help='是否测试VLAD性能')
    parser.add_argument('--num-test-samples', type=int, default=100, 
                        help='性能测试样本数')
    
    args = parser.parse_args()
    
    # Select device.
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # Create the VLAD trainer.
    trainer = DatabaseVLADTrainer(
        backbone_path=args.backbone_path,
        num_clusters=args.num_clusters,
        device=device
    )
    
    # Load RELLIS-3D dataloaders.
    print(f"加载RELLIS-3D {args.split} 数据集...")
    ontology_path = "/home/xie/Data/Rellis-3D/Rellis_3D_ontology/ontology.yaml"
    train_loader, val_loader, test_loader = get_rellis3d_ontology_dataloaders(
        data_root=args.data_root,
        ontology_path=ontology_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Select dataloader by split.
    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader
    
    print(f"使用 {args.split} 分割，包含 {len(dataloader.dataset)} 个样本")
    
    # Train VLAD.
    vlad_processor = trainer.train_vlad_from_database(
        data_loader=dataloader,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        sampling_strategy=args.sampling_strategy,
        cache_dir=args.cache_dir
    )
    
    # Optionally run the compression test.
    if args.test_performance:
        test_dataloader = val_loader if args.split != 'val' else test_loader
        performance_results = trainer.test_vlad_performance(
            vlad_processor, 
            test_dataloader, 
            num_test_samples=args.num_test_samples
        )
        
        print(f"\n性能测试结果:")
        print(f"平均压缩比: {performance_results['avg_compression_ratio']:.2f}")
    
    print(f"\nVLAD训练完成!")
    print(f"聚类数: {args.num_clusters}")
    print(f"训练样本数: {args.max_train_samples}")


if __name__ == "__main__":
    main()
