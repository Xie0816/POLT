"""RELLIS-3D ontology dataset utilities for offline DINO/VLAD scripts."""

import os
import torch
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class RELLIS3DOntologyDataset(Dataset):
    """RELLIS-3D semantic-segmentation dataset driven by ontology.yaml."""
    
    def __init__(self, data_root, split_file, ontology_path, transform=None, target_transform=None, image_size=(512, 512)):
        """Initialize dataset paths, transforms, and ontology mappings."""
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform
        
        # Load class metadata and label-ID mappings from ontology.yaml.
        self.num_classes, self.class_names, self.class_colors, self.original_to_continuous, self.continuous_to_original = self._load_ontology_info(ontology_path)
        
        # Read split entries such as train.lst, val.lst, or test.lst.
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label_path = parts[1]
                        self.samples.append((img_path, label_path))
        
        print(f"Loaded {len(self.samples)} samples from {split_file}")
        print(f"Loaded {self.num_classes} classes from {ontology_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_rel_path, label_rel_path = self.samples[idx]
        
        # Build absolute image and label paths.
        img_path = os.path.join(self.data_root, img_rel_path)
        label_path = os.path.join(self.data_root, label_rel_path)
        
        # Load RGB image and semantic label map.
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        
        # Apply image and target transforms separately.
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        # Map sparse raw labels to continuous IDs for training.
        label = self._map_labels_to_continuous(label)
        
        return image, label
    
    def _map_labels_to_continuous(self, label_tensor):
        """Map sparse ontology label IDs to a continuous 0..N-1 range."""
        # Create a tensor for mapped labels.
        mapped_label = torch.zeros_like(label_tensor)
        
        # Apply one mask per raw label ID.
        for original_id, continuous_id in self.original_to_continuous.items():
            mask = label_tensor == original_id
            mapped_label[mask] = continuous_id
        
        return mapped_label
    
    def _load_ontology_info(self, ontology_path):
        """Load class names, colors, and label-ID mappings from ontology.yaml."""
        with open(ontology_path, 'r') as f:
            ontology_data = yaml.safe_load(f)
        
        # First ontology entry maps class ID to class name.
        class_name_mapping = ontology_data[0]
        
        # Second ontology entry maps class ID to RGB color.
        class_color_mapping = ontology_data[1]
        
        # Build raw-ID <-> continuous-ID mappings.
        original_ids = sorted([int(k) for k in class_name_mapping.keys()])
        original_to_continuous = {}
        continuous_to_original = {}
        
        for continuous_id, original_id in enumerate(original_ids):
            original_to_continuous[original_id] = continuous_id
            continuous_to_original[continuous_id] = original_id
        
        num_classes = len(original_ids)
        
        # Build class metadata keyed by raw ontology ID.
        class_names = {}
        for original_id, name in class_name_mapping.items():
            original_id = int(original_id)
            continuous_id = original_to_continuous[original_id]
            class_names[original_id] = {
                "name": name,
                "value": original_id,
                "continuous_id": continuous_id
            }

        class_colors = {}
        for original_id, color in class_color_mapping.items():
            original_id = int(original_id)
            class_name = class_names[original_id]['name']
            if original_id in original_to_continuous:
                # Color format is [R, G, B].
                class_colors[class_name] = tuple(color)
        
        return num_classes, class_names, class_colors, original_to_continuous, continuous_to_original


def get_rellis3d_ontology_transforms(image_size=(512, 512)):
    """Return image and label transforms for RELLIS-3D ontology data."""
    
    # DINOv3 LVD-1689M weights use standard ImageNet evaluation normalization.
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Keep labels discrete by resizing with nearest-neighbor interpolation.
    def target_transform_func(label):
        label = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST)(label)
        label = torch.from_numpy(np.array(label)).long()
        return label
    
    return image_transform, target_transform_func


def get_rellis3d_ontology_dataloaders(data_root, ontology_path, batch_size=4, num_workers=4, image_size=(512, 512)):
    """
    Create RELLIS-3D dataloaders using ontology-based label remapping.
    
    Args:
        data_root: RELLIS-3D data root.
        ontology_path: Path to ontology.yaml.
        batch_size: Batch size.
        num_workers: DataLoader worker count.
        image_size: Output image size.
        
    Returns:
        Train, validation, and test dataloaders.
    """
    
    # Build shared transforms for images and labels.
    image_transform, target_transform = get_rellis3d_ontology_transforms(image_size)
    
    # Split files are expected directly under data_root.
    train_dataset = RELLIS3DOntologyDataset(
        data_root=data_root,
        split_file=os.path.join(data_root, 'train.lst'),
        ontology_path=ontology_path,
        transform=image_transform,
        target_transform=target_transform,
        image_size=image_size
    )
    
    val_dataset = RELLIS3DOntologyDataset(
        data_root=data_root,
        split_file=os.path.join(data_root, 'val.lst'),
        ontology_path=ontology_path,
        transform=image_transform,
        target_transform=target_transform,
        image_size=image_size
    )
    
    test_dataset = RELLIS3DOntologyDataset(
        data_root=data_root,
        split_file=os.path.join(data_root, 'test.lst'),
        ontology_path=ontology_path,
        transform=image_transform,
        target_transform=target_transform,
        image_size=image_size
    )
    
    # Construct dataloaders for each split.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_rellis3d_ontology_class_info(ontology_path):
    """Return RELLIS-3D class metadata from ontology.yaml."""
    import yaml
    
    with open(ontology_path, 'r') as f:
        ontology_data = yaml.safe_load(f)
    
    # First ontology entry maps class ID to class name.
    class_name_mapping = ontology_data[0]
    
    # Second ontology entry maps class ID to RGB color.
    class_color_mapping = ontology_data[1]
    
    # Build raw-ID <-> continuous-ID mappings.
    original_ids = sorted([int(k) for k in class_name_mapping.keys()])
    original_to_continuous = {}
    continuous_to_original = {}
    
    for continuous_id, original_id in enumerate(original_ids):
        original_to_continuous[original_id] = continuous_id
        continuous_to_original[continuous_id] = original_id
    
    num_classes = len(original_ids)
    
    # Build class metadata keyed by raw ontology ID.
    class_names = {}
    for original_id, name in class_name_mapping.items():
        original_id = int(original_id)
        continuous_id = original_to_continuous[original_id]
        class_names[original_id] = {
            "name": name,
            "value": original_id,
            "continuous_id": continuous_id
        }
    
    # Build class-color lookup keyed by class name.
    class_colors = {}
    for original_id, color in class_color_mapping.items():
        original_id = int(original_id)
        class_name = class_names[original_id]['name']
        if original_id in original_to_continuous:
            # Color format is [R, G, B].
            class_colors[class_name] = tuple(color)
    
    return num_classes, class_names, class_colors, original_to_continuous, continuous_to_original


def test_ontology_dataloader():
    """Smoke-test ontology dataloaders with local RELLIS-3D paths."""
    import numpy as np
    
    data_root = "/home/xie/Data/Rellis-3D"
    ontology_path = "/home/xie/Data/Rellis-3D/Rellis_3D_ontology/ontology.yaml"
    
    try:
        train_loader, val_loader, test_loader = get_rellis3d_ontology_dataloaders(
            data_root=data_root,
            ontology_path=ontology_path,
            batch_size=2,
            num_workers=0,  # Use a single-process loader for local tests.
            image_size=(256, 256)  # Use a smaller image for quick checks.
        )
        
        # Inspect one batch.
        for images, labels in train_loader:
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Labels unique values: {torch.unique(labels)}")
            break
            
        print("✓ Ontology DataLoader test passed!")
        
    except Exception as e:
        print(f"❌ Error testing Ontology DataLoader: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ontology_dataloader()
    
    # Inspect ontology metadata.
    ontology_path = "/home/xie/Data/Rellis-3D/Rellis_3D_ontology/ontology.yaml"
    num_classes, class_names, class_colors, original_to_continuous, continuous_to_original = get_rellis3d_ontology_class_info(ontology_path)
    
    print(f"\nRELLIS-3D 类别信息 (从 ontology.yaml 加载):")
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {class_names}")
    print(f"类别颜色: {class_colors}")
    print(f"原始到连续映射: {original_to_continuous}")
    print(f"连续到原始映射: {continuous_to_original}")
