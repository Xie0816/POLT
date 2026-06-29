"""Export class-centroid DINO features into POLT hierarchical memory format."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import setup_dino_project_paths

setup_dino_project_paths()
from model.memory.GPRMemoryForest import *

semantic_cost_dict = {
    "paved_road": {
        "asphalt":0.1, 
        "concrete":0.1
        },
    "unpaved_road":{
        "dirt": 0.8,
        "rubble":0.8,
        "mud":0.8
    },
    "grass_land":{
        "grass":0.9
    },
    "puddle":{
        "puddle":0.9
    },
    "fragile_barrier":{
        "bush": 1
    },
    "barrier":{
        "tree": 1,
        "pole": 1,
        "water": 1,
        "sky": 1,
        "vehicle":1,
        "object":1,
        "building":1,
        "log":1,
        "person":1,
        "fence":1,
        "barrier":1
    }
}

roughness_cost_dict = {
    "fragile_barrier":{
        "bush": 1
    },
    "barrier":{
        "tree": 1,
        "pole": 1,
        "sky": 1,
        "vehicle":1,
        "object":1,
        "building":1,
        "log":1,
        "person":1,
        "fence":1,
        "barrier":1
    }
}



def load_json(json_path):
    """Load a JSON file into a Python object."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def load_yaml(yaml_path):
    """Load a YAML file into a Python object."""
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

def export_featuresTobuffer(feat_json, save_path, cost_dict):
    """Convert exported class centroids into a GPRMemoryForest buffer."""

    memory_sampler = GPRMemoryForest()

    class_centroids = feat_json['class_centroids']
    features_per_class = feat_json['feature_statistics']['features_per_class']
    for class_key, centroids in class_centroids.items():
        access_nums = features_per_class[class_key]
        node_cost =None
        for meta_key, classes in cost_dict.items():
            if class_key in classes:
                node_cost = classes[class_key]
                break
        if node_cost == None:
            continue
        if meta_key not in memory_sampler.hierarchical_memory:
            memory_sampler._add_semantic_category(meta_key)
        for feat in  centroids["centroids"]:
            memory_sampler._add_semantic_cost_node(
                meta_key,
                feat,
                node_cost,
                access_nums=access_nums
            )
    memory_sampler.export_hierarchical_memory(export_dir=save_path)

def importFrombuffer(import_dir):
    """Load an exported hierarchical memory buffer for quick inspection."""

    memory_sampler = GPRMemoryForest()
    memory_sampler.import_hierarchical_memory(import_dir)

    print(memory_sampler.hierarchical_memory)

if __name__ == "__main__":

    ###     export test
    json_path = "outputs/dino_features_20260222_231247/dino_class_centroids.json"
    json_data = load_json(json_path)
    save_path = "mem_buffer/meta_memory"
    export_featuresTobuffer(json_data, save_path, semantic_cost_dict)

    ###     import test
    import_dir = "mem_buffer/meta_memory"
    importFrombuffer(import_dir)



    # salon
    ###     export test
    # json_path = "outputs/dino_features_20260222_231247/dino_class_centroids.json"
    # json_data = load_json(json_path)

    # save_path = "data_buffer_salon/data_buffer"
    # export_featuresTobuffer(json_data, save_path, roughness_cost_dict)

    # ###     import test
    # import_dir = "data_buffer"
    # importFrombuffer(import_dir)
