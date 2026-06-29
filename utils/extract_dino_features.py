"""Offline DINO feature extraction and class-centroid export utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import setup_dino_project_paths

setup_dino_project_paths()
from utils.rellis3d_data.rellis3d import get_rellis3d_ontology_dataloaders, get_rellis3d_ontology_class_info
from dinov3.hub.backbones import dinov3_vits16plus


class DINOFeatureExtractor:
    """DINO feature extractor for RELLIS-3D style datasets."""
    
    def __init__(self, backbone_path, device='cuda', ontology_path=None):
        self.device = device
        
        print(f"Loading DINO backbone from: {backbone_path}")
        self.backbone = dinov3_vits16plus(
            pretrained=True,
            weights=backbone_path
        ).to(device)
        
        self.backbone.eval()
        
        if ontology_path is None:
            ontology_path = "/home/xie/Data/Rellis-3D/Rellis_3D_ontology/ontology.yaml"
        
        self.num_classes, self.class_names, self.class_colors, _, _ = get_rellis3d_ontology_class_info(ontology_path)
        
        print(f"Feature extractor initialized with {self.num_classes} classes")
    
    def extract_features(self, images):
        """Extract patch tokens and feature-map shape from a batch of images."""
        with torch.no_grad():
            tokens, (H, W) = self.backbone.prepare_tokens_with_masks(images)

            features_dict = self.backbone.forward_features(images)

            patch_features = features_dict["x_norm_patchtokens"]

        return patch_features, tokens, (H, W)

    
    def compute_class_centroid_features(self, features, masks, k=5):
        """Compute representative feature centroids for each semantic class."""
        features = features.to(self.device)
        masks = masks.to(self.device)
        
        # Convert spatial feature maps to flattened patch descriptors.
        if len(features.shape) == 4:
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
            masks = masks.reshape(B, H*W)
        
        B, N, C = features.shape
        
        class_centroids = {}
        
        for class_value in self.class_names.values():
            class_id = class_value['continuous_id']
            # Collect descriptors for this class across the current batch.
            all_class_features = []
            
            for batch_idx in range(B):
                # Locate patch positions belonging to this semantic class.
                class_mask = (masks[batch_idx] == class_id)  # (N,)
                
                if class_mask.sum() == 0:
                    # Skip images where this class is absent.
                    continue
                
                # Gather descriptors for this class.
                class_feat_indices = torch.where(class_mask)[0]
                class_features_batch = features[batch_idx, class_feat_indices]  # (M, C)
                
                if len(class_features_batch) == 0:
                    continue
                
                # Append CPU numpy arrays to reduce GPU memory pressure.
                all_class_features.append(class_features_batch.cpu().numpy())
            
            if not all_class_features:
                # Keep an empty entry when no descriptors are available.
                class_centroids[class_id] = []
                continue
            
            # Merge all descriptors before clustering.
            all_class_features = np.vstack(all_class_features)  # (total_M, C)
            
            if len(all_class_features) < k:
                # Use all available descriptors if fewer than k exist.
                k_actual = len(all_class_features)
            else:
                k_actual = k
            
            # Use K-means to find k representative centroids.
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
            kmeans.fit(all_class_features)
            
            # Store centroids and cluster sizes.
            centroids = kmeans.cluster_centers_  # (k_actual, C)
            
            class_centroids[class_id] = []
            for i, centroid in enumerate(centroids):
                class_centroids[class_id].append({
                    'centroid': centroid,
                    'cluster_id': i,
                    'num_samples_in_cluster': np.sum(kmeans.labels_ == i)
                })
        
        return class_centroids


def visualize_features_with_tsne(all_class_raw_features, class_names, class_colors, output_dir, max_samples=5000):
    """Visualize sampled class descriptors with t-SNE."""
    
    # Merge descriptors and labels across all classes.
    all_features = []
    all_labels = []
    
    for class_id, features_list in all_class_raw_features.items():
        if features_list:
            # Merge all descriptors collected for this class.
            class_features = np.vstack(features_list)
            all_features.append(class_features)
            all_labels.extend([class_id] * len(class_features))
    
    if not all_features:
        print("No features to visualize")
        return
    
    # Merge all classes into one t-SNE input matrix.
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    # Sample descriptors when the set is too large for fast t-SNE.
    original_sample_count = len(all_features)
    if original_sample_count > max_samples:
        print(f"Sampling {max_samples} from {original_sample_count} features for t-SNE visualization")
        # Random subsampling keeps visualization runtime bounded.
        indices = np.random.choice(original_sample_count, max_samples, replace=False)
        all_features = all_features[indices]
        all_labels = all_labels[indices]
    
    print(f"Visualizing {len(all_features)} features with t-SNE...")
    
    # Run t-SNE reduction to two dimensions.
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(all_features)
    
    # Draw one scatter group per semantic class.
    plt.figure(figsize=(12, 10))
    
    # Use ontology colors where available.
    unique_labels = np.unique(all_labels)
    
    for i, class_id in enumerate(unique_labels):
        mask = all_labels == class_id
        class_name = class_names.get(class_id, {}).get('name', f'{class_id}')
        
        class_color_info = class_colors.get(class_id, {})
        # Convert RGB from 0-255 to 0-1 for Matplotlib.
        color_rgb = [c/255.0 for c in class_color_info]

        
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[color_rgb], label=class_name, alpha=0.7, s=10)
    
    plt.title('t-SNE Visualization of DINO Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save full descriptor visualization.
    tsne_path = os.path.join(output_dir, 'tsne_visualization.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to: {tsne_path}")
    
    # Create a second view that shows class centroids only.
    plt.figure(figsize=(12, 10))
    
    # Average t-SNE points per class as representative markers.
    for i, class_id in enumerate(unique_labels):
        mask = all_labels == class_id
        if np.sum(mask) > 0:
            class_features = features_2d[mask]
            # Compute the class center in t-SNE space.
            centroid = np.mean(class_features, axis=0)
            class_name = class_names.get(class_id, {}).get('name', f'{class_id}')
            
            class_color_info = class_colors.get(class_id, {})
            # Convert RGB from 0-255 to 0-1 for Matplotlib.
            color_rgb = [c/255.0 for c in class_color_info]
            
            plt.scatter(centroid[0], centroid[1], 
                       c=[color_rgb], label=class_name, s=200, marker='*', edgecolors='black')
    
    plt.title('t-SNE Visualization - Class Centroids')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save centroid-only visualization.
    centroid_path = os.path.join(output_dir, 'tsne_centroids.png')
    plt.savefig(centroid_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE centroids visualization saved to: {centroid_path}")


def extract_dataset_features(data_loader, feature_extractor, k=5, output_dir=None):
    """Extract dataset-level DINO descriptors and cluster them by class."""
    
    if output_dir is None:
        output_dir = f'./outputs/dino_features_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect raw descriptors per class before dataset-level clustering.
    all_class_raw_features = {class_value['name']: [] for class_value in feature_extractor.class_names.values()}
    
    pbar = tqdm(data_loader, desc='Extracting features')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(feature_extractor.device)
        masks = masks.to(feature_extractor.device)
        
        # Convert masks from (B, 1, H, W) to (B, H, W) LongTensor.
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        masks = masks.long()
        
        # Extract DINO patch descriptors.
        patch_features, tokens, (H, W) = feature_extractor.extract_features(images)
        
        # Resize masks to the DINO patch grid.
        B, N, C = patch_features.shape
        resized_masks = F.interpolate(
            masks.unsqueeze(1).float(), 
            size=(H, W), 
            mode='nearest'
        ).squeeze(1).long()
        
        # Align flattened masks with patch descriptor order.
        if len(patch_features.shape) == 3:
            B, N, C = patch_features.shape
            features_flat = patch_features
            masks_flat = resized_masks.flatten(1)
        
        # Collect raw descriptors for every class in the batch.
        for batch_idx in range(B):
            for class_value in feature_extractor.class_names.values():
                class_id = class_value['continuous_id']
                class_name = class_value['name']
                # Locate patch positions belonging to this class.
                class_mask = (masks_flat[batch_idx] == class_id)  # (N,)
                
                if class_mask.sum() == 0:
                    # Skip classes absent from this image.
                    continue
                
                # Gather class descriptors.
                class_feat_indices = torch.where(class_mask)[0]
                class_features_batch = features_flat[batch_idx, class_feat_indices]  # (M, C)
                
                if len(class_features_batch) == 0:
                    continue
                
                # Store descriptors on CPU to keep GPU memory bounded.
                all_class_raw_features[class_name].append(class_features_batch.cpu().numpy())
        
        pbar.set_postfix({
            'Batch': batch_idx,
            'Total Features': sum(len(features) for features in all_class_raw_features.values())
        })
    
    print("All features collected, starting dataset-level clustering...")
    
    # Cluster descriptors per class at dataset level.
    dataset_class_centroids = {}
    
    for class_value in feature_extractor.class_names.values():
        class_id = class_value['continuous_id']
        class_name = class_value['name']
        if not all_class_raw_features[class_name]:
            # Keep an empty entry when no descriptors are available.
            dataset_class_centroids[class_name] = []
            continue
        
        # Merge all descriptors for this class.
        all_class_features = np.vstack(all_class_raw_features[class_name])  # (total_M, C)
        
        if len(all_class_features) < k:
            # Use all available descriptors if fewer than k exist.
            k_actual = len(all_class_features)
        else:
            k_actual = k
        
        # Bound clustering runtime by sampling large classes.
        max_samples_per_class = 10000  # Maximum samples per class.
        if len(all_class_features) > max_samples_per_class:
            print(f"Class {class_name}: Sampling {max_samples_per_class} from {len(all_class_features)} features")
            # Randomly sample descriptors for K-means.
            indices = np.random.choice(len(all_class_features), max_samples_per_class, replace=False)
            sampled_features = all_class_features[indices]
        else:
            sampled_features = all_class_features
        
        # Use faster K-means settings for large descriptor sets.
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(
            n_clusters=k_actual, 
            random_state=42, 
            n_init=3,
            max_iter=100,
            algorithm='elkan'
        )
        kmeans.fit(sampled_features)
        
        # Store centroids and metadata.
        centroids = kmeans.cluster_centers_  # (k_actual, C)
        
        dataset_class_centroids[class_name] = []
        for i, centroid in enumerate(centroids):
            dataset_class_centroids[class_name].append({
                'centroid': centroid,
                'cluster_id': i,
                'num_samples_in_cluster': np.sum(kmeans.labels_ == i),
                'total_samples': len(sampled_features),
                'original_total_samples': len(all_class_features)
            })
    
    # Serialize centroids and extraction metadata.
    output_data = {
        'extraction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_classes': feature_extractor.num_classes,
        'class_names': feature_extractor.class_names,
        'extraction_parameters': {},
        'feature_statistics': {},
        'class_centroids': {}
    }
    
    for class_name, centroids_list in dataset_class_centroids.items():
        if centroids_list:
            # Convert centroid lists into arrays before JSON encoding.
            centroids_array = np.array([centroid['centroid'] for centroid in centroids_list])
            
            output_data['class_centroids'][class_name] = {
                'centroids': centroids_array,
                'metadata': [{
                    'cluster_id': centroid['cluster_id'],
                    'num_samples_in_cluster': centroid['num_samples_in_cluster'],
                    'total_samples': centroid['total_samples']
                } for centroid in centroids_list]
            }

    output_data['extraction_parameters'] = {
            'k': k,
            'backbone': 'dinov3_vits16plus',
            'dataset': 'RELLIS-3D',
            'clustering_level': 'dataset_level'
        }
    
    output_data['feature_statistics'] = {
        'total_features_collected': sum(len(features) for features in all_class_raw_features.values()),
        'features_per_class': {class_name: len(features) for class_name, features in all_class_raw_features.items()}
    }
    
    # Save as JSON.
    output_path = os.path.join(output_dir, 'dino_class_centroids.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"Dataset-level class centroids saved to: {output_path}")

    
    # Generate optional t-SNE diagnostics.
    print("Generating t-SNE visualizations...")
    visualize_features_with_tsne(all_class_raw_features, feature_extractor.class_names, feature_extractor.class_colors, output_dir, max_samples=5000)
    
    return dataset_class_centroids


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
    parser = argparse.ArgumentParser(description='Extract DINO features and compute class-wise similarity')
    parser.add_argument('--backbone-path', type=str, default='weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth', help='Path to backbone weights')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')  
    parser.add_argument('--image-size', type=int, default=(600,960), help='Image size')  
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--data-root', type=str, default="/home/xie/Data/Rellis-3D", help='RELLIS-3D data root directory')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for features')
    parser.add_argument('--k', type=int, default=1, help='Number of top similar features to extract per class')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='train', help='Dataset split to use')
    
    args = parser.parse_args()
    
    # Select device.
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create the DINO feature extractor.
    feature_extractor = DINOFeatureExtractor(
        backbone_path=args.backbone_path,
        device=device
    )
    
    # Load the selected RELLIS-3D split.
    print(f"Loading RELLIS-3D {args.split} dataset...")
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
    
    print(f"Using {args.split} split with {len(dataloader.dataset)} samples")
    
    # Extract and cluster features.
    dataset_class_centroids = extract_dataset_features(
        data_loader=dataloader,
        feature_extractor=feature_extractor,
        k=args.k,
        output_dir=args.output_dir
    )
    
    # Print extraction summary.
    print(f"\nFeature extraction completed!")
    total_centroids = sum(len(centroids) for centroids in dataset_class_centroids.values())
    print(f"Total centroids extracted: {total_centroids}")
    
    for class_value in feature_extractor.class_names.values():
        class_id = class_value['value']
        class_name = class_value['name']
        num_centroids = len(dataset_class_centroids[class_name])
        print(f"Class {class_id} ({class_name}): {num_centroids} centroids")


if __name__ == "__main__":
    main()
