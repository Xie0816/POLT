"""Merge manually exported memory nodes into an existing POLT memory buffer."""

import os

from model.memory.GPRMemoryForest import GPRMemoryForest
from model.vision.dinov3_infer import Dinov3Infer


if __name__ == "__main__":
    # TODO: expose these paths as CLI arguments if this helper becomes public.
    root = "/home/xie/Data/Terrain_Dataset/puddle7" 
    mem_buffer_dir = "mem_buffer/baotou"
    import_dir = "mem_buffer/click_add/"
    
    dino_sampler = Dinov3Infer(use_vlad=True, vlad_clusters=32)
    mf_sampler = GPRMemoryForest()
    if mem_buffer_dir:
        mf_sampler._mf_init(mem_buffer_dir, vlad=dino_sampler.vlad_processor)

    for json_file in os.listdir(import_dir):
        json_path = os.path.join(import_dir, json_file)

        mem_info = mf_sampler.import_node(json_path, vlad=dino_sampler.vlad_processor)

        # print(mem_info.get('mechanism_type', None))

    mf_sampler.export_hierarchical_memory("mem_buffer/baotou_puddle_new")
