import json
import os
from collections import Counter

def analyze_object_counts(dataset_path):
    duplicate_count = 0
    total_label_counts = Counter()  # New global counter

    # Check if path exists first
    if not os.path.exists(dataset_path):
        print(f"Error: The path '{dataset_path}' does not exist.")
        return 0, Counter()

    for scene_path in os.listdir(dataset_path):
        # Skip non-json files
        if not scene_path.endswith(".json"):
            continue

        local_labels = {}
        
        try:
            with open(os.path.join(dataset_path, scene_path), "r") as f:
                data = json.load(f).get("nodes", [])
                
                for node in data:
                    attributes = node.get("attributes", {})
                    # Ensure attributes is actually a dict before accessing .get
                    if attributes:
                        label = attributes.get("label")
                        
                        if label:
                            # 1. Track for local duplicates (original logic)
                            local_labels[label] = local_labels.get(label, 0) + 1
                            
                            # 2. Track for global total (new logic)
                            total_label_counts[label] += 1
            
            # Count duplicates for this specific file
            for count in local_labels.values():
                if count > 1:
                    duplicate_count += 1
                    
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Skipping file {scene_path} due to error: {e}")

    return duplicate_count, total_label_counts

if __name__ == "__main__":
    dataset_path = "/home/adamov/fergus/GRID/dataset/GRID_Dataset-master/Mini_Dataset/data/scene.1.graphs"
    
    dupes, total_counts = analyze_object_counts(dataset_path)
    
    print(f"Total duplicate events found: {dupes}")
    print("-" * 30)
    print("Total count of each label across all files:")
    # Print sorted by count (highest first)
    for label, count in total_counts.most_common():
        print(f"{label}: {count}")