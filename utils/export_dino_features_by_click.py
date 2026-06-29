"""Interactive tool for exporting manually selected DINO patch features."""

import torch
import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import setup_dino_project_paths

setup_dino_project_paths()
from model.memory.GPRMemoryForest import *
from utils.extract_dino_features import DINOFeatureExtractor

class ClickExporter:
    """OpenCV click-based exporter for one image's DINO patch features."""

    def __init__(self, backbone_path, output_path, device='cuda'):
        self.extractor = DINOFeatureExtractor(backbone_path=backbone_path, device=device)
        self.memory_manager = GPRMemoryForest()
        
        self.current_patch_features = None
        self.grid_size = None
        self.img_display = None
        self.combined_view = None
        self.output_path = output_path
        self.img_h, self.img_w = 0, 0

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _get_feature_visualization(self, features, hp, wp):
        """Project high-dimensional patch features to RGB via PCA."""
        # features shape: (N, C) -> (N, 384)
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features)
        
        # Normalize PCA channels to displayable uint8 RGB.
        pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0) + 1e-6)
        pca_features = (pca_features * 255).astype(np.uint8)
        
        feat_map = pca_features.reshape(hp, wp, 3)
        # Nearest-neighbor resizing preserves visible patch boundaries.
        feat_map_resized = cv2.resize(feat_map, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        return feat_map_resized

    def load_image(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("图像加载失败")
            return
        
        self.img_h, self.img_w = img_bgr.shape[:2]
        self.img_display = img_bgr.copy()
        
        # Match the training/inference resolution used by the DINO utilities.
        input_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float().unsqueeze(0).to(self.extractor.device) / 255.0
        input_tensor = torch.nn.functional.interpolate(input_tensor, size=(600, 960))
        
        patch_features, _, (H_p, W_p) = self.extractor.extract_features(input_tensor)
        self.current_patch_features = patch_features.squeeze(0).cpu().numpy()
        self.grid_size = (H_p, W_p)

        feat_viz = self._get_feature_visualization(self.current_patch_features, H_p, W_p)
        
        # Display raw image and PCA feature view side by side.
        self.combined_view = np.hstack((self.img_display, feat_viz))
        return self.combined_view

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map clicks from either side of the combined view back to image coordinates.
            click_x = x if x < self.img_w else x - self.img_w
            
            hp, wp = self.grid_size
            col = int(click_x / self.img_w * wp)
            row = int(y / self.img_h * hp)
            patch_idx = row * wp + col
            
            clicked_feat = self.current_patch_features[patch_idx]
            
            # Export the clicked patch as one memory node.
            temp_node = SemanticCostNode(
                semantic_features=clicked_feat,
                proprio_cost=0.0 
            )
            node_dict = self.memory_manager._semantic_cost_node_to_dict(temp_node)
            
            timestamp = datetime.now().strftime("%H%M%S")
            filename = os.path.join(self.output_path, f"node_{row}_{col}_{timestamp}.json")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(node_dict, f, indent=2, ensure_ascii=False)
            
            print(f"成功导出节点: {filename} (Row:{row}, Col:{col})")
            
            # Visual feedback on both the raw image and feature view.
            temp_view = self.combined_view.copy()
            cv2.circle(temp_view, (click_x, y), 8, (0, 0, 255), -1)              # Mark source image.
            cv2.circle(temp_view, (click_x + self.img_w, y), 8, (0, 0, 255), -1) # Mark feature view.
            cv2.imshow("DINO View: Image (Left) | Feature (Right)", temp_view)

def main():
    BACKBONE = 'weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
    IMG_PATH = '/home/xie/Data/Terrain_Dataset/puddle12/image_front/1727059532357.png'
    output_path = 'mem_buffer/click_add'

    exporter = ClickExporter(BACKBONE, output_path)
    combined_img = exporter.load_image(IMG_PATH)
    
    if combined_img is None:
        return

    window_name = "DINO View: Image (Left) | Feature (Right)"
    
    # Configure a resizable OpenCV window for interactive annotation.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    
    # Read stitched-view size.
    h, w = combined_img.shape[:2]
    
    # Use a reasonable initial width and preserve aspect ratio.
    display_width = 1280
    display_height = int(h * (display_width / w))
    cv2.resizeWindow(window_name, display_width, display_height)
    # --------------------------

    cv2.setMouseCallback(window_name, exporter.on_mouse_click)
    
    print(">>> 操作提示:")
    print(f"1. 原始拼接尺寸: {w}x{h}，已自动缩放窗口至 {display_width}x{display_height} 以适应屏幕。")
    print("2. 您可以手动拉伸窗口边缘来调整大小。")
    print("3. 点击任意位置导出 JSON 节点，按 'q' 退出。")
    
    while True:
        cv2.imshow(window_name, combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
