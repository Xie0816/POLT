"""VLAD feature encoder and clustering utilities for DINO patch descriptors."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import fast_pytorch_kmeans as fpk
import einops as ein
from typing import Union, List

class VLAD:
    """VLAD encoder for local patch descriptors.

    Parameters:
        num_clusters: Number of cluster centers.
        desc_dim: Descriptor dimension; inferred during ``fit`` when omitted.
        intra_norm: Normalize each cluster residual before concatenation.
        norm_descs: Normalize descriptors before training and assignment.
        dist_mode: KMeans distance mode, usually ``euclidean`` or ``cosine``.
        vlad_mode: Descriptor assignment mode, ``soft`` or ``hard``.
        soft_temp: Temperature used by soft assignment.
        cache_dir: Optional directory for cluster-center and residual caches.
        override_cache: Recompute cluster centers even when cache exists.
    """
    def __init__(self, num_clusters: int,
                desc_dim: Union[int, None]=None,
                intra_norm: bool=True, norm_descs: bool=True,
                dist_mode: str="cosine", vlad_mode: str="soft",
                soft_temp: float=10.0,
                cache_dir: Union[str,None]=None,
                override_cache: bool=False) -> None:
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = dist_mode
        self.vlad_mode = str(vlad_mode).lower()
        assert self.vlad_mode in ['soft', 'hard']
        self.soft_temp = soft_temp
        # Populated after fitting or loading cached centers.
        self.c_centers = None
        self.kmeans = None

        self.cache_dir = cache_dir
        self.override_cache = override_cache
        
        if self.cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(self.cache_dir))
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                print(f"创建缓存目录: {self.cache_dir}")
            else:
                print(f"警告: 缓存目录已存在: {self.cache_dir}")
        else:
            print("VLAD缓存已禁用")

    def can_use_cache_vlad(self):
        """Return whether cached cluster centers are available."""
        if self.cache_dir is None:
            return False
        if not os.path.exists(self.cache_dir):
            return False
        if os.path.exists(f"{self.cache_dir}/c_centers.pt"):
            return True
        else:
            return False

    def fit(self, train_descs: Union[np.ndarray, torch.Tensor, None] = None):
        """Fit or load the VLAD visual vocabulary."""
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        
        if self.can_use_cache_vlad() and not self.override_cache:
            print("使用缓存的聚类中心")
            self.c_centers = torch.load(f"{self.cache_dir}/c_centers.pt").cuda()
            self.kmeans.centroids = self.c_centers
            if self.desc_dim is None:
                self.desc_dim = self.c_centers.shape[1]
                print(f"描述符维度设置为 {self.desc_dim}")
        else:
            if train_descs is None:
                raise ValueError("未提供训练描述符")
            if type(train_descs) == np.ndarray:
                train_descs = torch.from_numpy(train_descs).to(torch.float32)
            if self.desc_dim is None:
                self.desc_dim = train_descs.shape[1]
            if self.norm_descs:
                train_descs = F.normalize(train_descs)
            self.kmeans.fit(train_descs)
            self.c_centers = self.kmeans.centroids
            if self.cache_dir is not None:
                print("缓存聚类中心")
                torch.save(self.c_centers, f"{self.cache_dir}/c_centers.pt")

    def generate_res_vec(self,
                query_descs: Union[np.ndarray, torch.Tensor],
                cache_id: Union[str, None]=None) -> torch.Tensor:
        """Generate residuals from query descriptors to every cluster center."""
        assert self.kmeans is not None
        assert self.c_centers is not None
        
        # Keep cached centers on the same device as query descriptors.
        if self.c_centers.device != query_descs.device:
            self.c_centers = self.c_centers.to(query_descs.device)
            if self.kmeans is not None:
                self.kmeans.centroids = self.c_centers
        
        if cache_id is not None and self.can_use_cache_vlad() and \
                os.path.isfile(f"{self.cache_dir}/{cache_id}_r.pt"):
            residuals = torch.load(f"{self.cache_dir}/{cache_id}_r.pt")
            if residuals.device != query_descs.device:
                residuals = residuals.to(query_descs.device)
        else:
            if type(query_descs) == np.ndarray:
                query_descs = torch.from_numpy(query_descs).to(torch.float32)
            if query_descs.device != self.c_centers.device:
                query_descs = query_descs.to(self.c_centers.device)
            if self.norm_descs:
                query_descs = F.normalize(query_descs)
            residuals = ein.rearrange(query_descs, "q d -> q 1 d") \
                    - ein.rearrange(self.c_centers, "c d -> 1 c d")
            if cache_id is not None and self.can_use_cache_vlad():
                cid_dir = f"{self.cache_dir}/{os.path.split(cache_id)[0]}"
                if not os.path.isdir(cid_dir):
                    os.makedirs(cid_dir)
                    print(f"创建目录: {cid_dir}")
                torch.save(residuals, f"{self.cache_dir}/{cache_id}_r.pt")
        return residuals

    def generate(self, query_descs: Union[np.ndarray, torch.Tensor],
                cache_id: Union[str, None]=None, reduce: bool=False) -> torch.Tensor:
        """Generate a normalized VLAD vector using the AnyLoc-style formulation."""
        residuals = self.generate_res_vec(query_descs, cache_id)
        
        # Unnormalized VLAD vector with shape [num_clusters * desc_dim].
        un_vlad = torch.zeros(self.num_clusters * self.desc_dim, device=residuals.device)
        if reduce:
            reduced_vlad = torch.zeros((self.num_clusters, self.desc_dim), device=residuals.device)

        if self.vlad_mode == 'hard':
            # Hard assignment stores one cluster label per descriptor.
            if cache_id is not None and self.can_use_cache_vlad() \
                    and os.path.isfile(f"{self.cache_dir}/{cache_id}_l.pt"):
                labels = torch.load(f"{self.cache_dir}/{cache_id}_l.pt")
            else:
                # Convert query descriptors to tensors on the cluster-center device.
                if isinstance(query_descs, np.ndarray):
                    query_descs = torch.from_numpy(query_descs).to(torch.float32)
                if query_descs.device != self.c_centers.device:
                    query_descs = query_descs.to(self.c_centers.device)
                labels = self.kmeans.predict(query_descs)   # [q]
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(labels, f"{self.cache_dir}/{cache_id}_l.pt")
            
            # Aggregate residuals cluster by cluster.
            used_clusters = set(labels.cpu().numpy())
            for k in used_clusters:
                # Sum descriptor residuals assigned to the cluster: [q', d] -> [d].
                cd_sum = residuals[labels==k, k].sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
                if reduce:
                    reduced_vlad[k] = cd_sum
        else:
            # Soft assignment uses cosine similarity: 1 means near, -1 means far.
            if cache_id is not None and self.can_use_cache_vlad() \
                    and os.path.isfile(f"{self.cache_dir}/{cache_id}_s.pt"):
                soft_assign = torch.load(f"{self.cache_dir}/{cache_id}_s.pt")
            else:
                # Convert query descriptors to tensors on the cluster-center device.
                if isinstance(query_descs, np.ndarray):
                    query_descs = torch.from_numpy(query_descs).to(torch.float32)
                if query_descs.device != self.c_centers.device:
                    query_descs = query_descs.to(self.c_centers.device)
                cos_sims = F.cosine_similarity(
                        ein.rearrange(query_descs, "q d -> q 1 d"),
                        ein.rearrange(self.c_centers, "c d -> 1 c d"),
                        dim=2)
                soft_assign = F.softmax(self.soft_temp * cos_sims, dim=1)
                if cache_id is not None and self.can_use_cache_vlad():
                    torch.save(soft_assign, f"{self.cache_dir}/{cache_id}_s.pt")
            
            # Soft assignment scores act as probabilities: [q, c].
            for k in range(0, self.num_clusters):
                w = ein.rearrange(soft_assign[:, k], "q -> q 1 1")
                # Weighted residual sum for cluster k.
                cd_sum = ein.rearrange(w * residuals, "q c d -> (q c) d").sum(dim=0)  # [d]
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        
        # Normalize the final VLAD vector following the AnyLoc implementation.
        if reduce:
            n_vlad = F.normalize(torch.sum(reduced_vlad, dim=0), dim=0)
            return n_vlad
        else:
            n_vlad = F.normalize(un_vlad, dim=0)
            return n_vlad

    def generate_multi(self,
            multi_query: Union[np.ndarray, torch.Tensor, list],
            cache_ids: Union[List[str], None]=None) -> Union[torch.Tensor, list]:
        """
        Generate VLAD vectors for multiple images.

        Args:
            multi_query: Descriptors with shape ``[n_imgs, n_kpts, d]``.
            cache_ids: Optional cache IDs aligned with ``multi_query``.
        """
        if cache_ids is None:
            cache_ids = [None] * len(multi_query)
        res = [self.generate(q, c) for (q, c) in zip(multi_query, cache_ids)]
        try:
            res = torch.stack(res)
        except TypeError:
            try:
                res = np.stack(res)
            except TypeError:
                pass
        return res

    def get_nearest_cluster_id(self, query_desc: Union[np.ndarray, torch.Tensor], 
                              return_distance: bool = False, 
                              return_top_k: int = 1,
                              feature_type: str = "cluster_rep") -> Union[int, tuple]:
        """
        Return the nearest VLAD cluster ID for a descriptor representation.

        ``feature_type`` can be ``raw``, ``vlad``, or ``cluster_rep``. The
        function can also return distance values or the top-k nearest clusters.
        """
        assert self.kmeans is not None
        assert self.c_centers is not None
        assert feature_type in ["raw", "vlad", "cluster_rep"], \
            f"feature_type must be 'raw', 'vlad', or 'cluster_rep'; got {feature_type}"
        
        # Convert input to a tensor on the cluster-center device.
        if isinstance(query_desc, np.ndarray):
            query_desc = torch.from_numpy(query_desc).to(torch.float32)
        
        if query_desc.device != self.c_centers.device:
            query_desc = query_desc.to(self.c_centers.device)
        
        # Add a batch dimension for single descriptors.
        if len(query_desc.shape) == 1:
            query_desc = query_desc.unsqueeze(0)
        
        if feature_type == "cluster_rep":
            # Cluster representation is either one-hot or probability-like.
            if self.vlad_mode == 'hard':
                max_vals, max_idxs = torch.max(query_desc, dim=1)
            else:
                max_vals, max_idxs = torch.max(query_desc, dim=1)
            
            if return_top_k == 1:
                cluster_id = int(max_idxs[0])
                if return_distance:
                    # For cluster_rep, distance is interpreted as 1 - confidence.
                    distance = 1 - float(max_vals[0]) if max_vals[0] <= 1 else 0.0
                    return cluster_id, distance
                else:
                    return cluster_id
            else:
                # Return top-k most confident cluster IDs.
                topk_vals, topk_idxs = torch.topk(query_desc, k=return_top_k, dim=1, largest=True)
                cluster_ids = topk_idxs[0].cpu().numpy().tolist()
                
                if return_distance:
                    # Convert confidence to distance-like values.
                    distances_list = [(1 - float(val)) if val <= 1 else 0.0 for val in topk_vals[0].cpu().numpy()]
                    return cluster_ids, distances_list
                else:
                    return cluster_ids
        else:
            # Handle raw descriptors or reduced VLAD descriptors.
            if self.norm_descs:
                query_desc = F.normalize(query_desc)
            
            # Compute distance to all cluster centers.
            if self.mode == "cosine":
                cos_similarity = F.cosine_similarity(
                    ein.rearrange(query_desc, "n d -> n 1 d"),
                    ein.rearrange(self.c_centers, "c d -> 1 c d"),
                    dim=2
                )
                distances = 1 - cos_similarity
            else:
                distances = torch.cdist(query_desc, self.c_centers, p=2)
            
            # Return nearest or top-k cluster IDs.
            if return_top_k == 1:
                min_dist, min_idx = torch.min(distances, dim=1)
                cluster_id = int(min_idx[0])
                
                if return_distance:
                    return cluster_id, float(min_dist[0])
                else:
                    return cluster_id
            else:
                topk_dist, topk_idx = torch.topk(distances, k=return_top_k, dim=1, largest=False)
                cluster_ids = topk_idx[0].cpu().numpy().tolist()
                
                if return_distance:
                    distances_list = topk_dist[0].cpu().numpy().tolist()
                    return cluster_ids, distances_list
                else:
                    return cluster_ids

    def get_cluster_assignments(self, query_descs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Return soft cluster-assignment probabilities with shape ``[N, K]``."""
        assert self.c_centers is not None, "VLAD is not initialized; call fit() first"
        
        # Convert format and align device.
        if isinstance(query_descs, np.ndarray):
            query_descs = torch.from_numpy(query_descs).to(torch.float32)
        if query_descs.device != self.c_centers.device:
            query_descs = query_descs.to(self.c_centers.device)
            
        # Normalize descriptors before cosine assignment.
        if self.norm_descs:
            query_descs = F.normalize(query_descs)
            
        # Compute cosine similarity [N, K].
        # query: [N, 1, D], centers: [1, K, D] -> sum(dim=2) -> [N, K]
        cos_sims = F.cosine_similarity(
                query_descs.unsqueeze(1),
                self.c_centers.unsqueeze(0),
                dim=2)
        
        # Apply temperature and softmax.
        soft_assign = F.softmax(self.soft_temp * cos_sims, dim=1)
        
        return soft_assign

def create_vlad_processor(num_clusters=32, cache_dir=None):
    """Create and fit a VLAD processor from cached or provided centers."""
    vlad_sampler = VLAD(num_clusters=num_clusters, cache_dir=cache_dir)
    vlad_sampler.fit()
    return vlad_sampler


if __name__ == "__main__":
    # Local smoke test.
    import torch
    
    # Create random patch descriptors.
    batch_size = 2
    num_patches = 100
    feature_dim = 384
    
    # Training descriptors.
    train_features = torch.randn(500, feature_dim).cuda()
    
    # Query descriptors.
    test_features = torch.randn(batch_size, num_patches, feature_dim).cuda()
    
    # Create and fit the VLAD processor.
    vlad_processor = create_vlad_processor(num_clusters=64)
    
    vlad_processor.fit(train_features)
    
    # Convert query descriptors to VLAD vectors.
    vlad_features = vlad_processor.transform(test_features, reduce=True)
    
    print(f"输入特征形状: {test_features.shape}")
    print(f"VLAD特征形状: {vlad_features.shape}")
    print(f"降维比例: {test_features.shape[-1] * num_patches} -> {vlad_features.shape[-1]}")
