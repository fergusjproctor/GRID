import json
import os
from collections import Counter
import matplotlib.pyplot as plt

def analyze_scenegraph_stats(folder_paths, output_file="scenegraph_stats.png"):
    # Initialize global counters for ALL folders
    total_node_types = Counter()
    total_edge_types = Counter()
    
    total_files_processed = 0

    # --- 1. DATA COLLECTION ---
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' not found. Skipping.")
            continue

        print(f"Scanning folder: {folder_path} ...")

        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Count Node Types
                        if "nodes" in data and isinstance(data["nodes"], list):
                            for node in data["nodes"]:
                                n_type = node.get("type", "unknown_node_type")
                                total_node_types[n_type] += 1

                        # Count Edge Types
                        if "edges" in data and isinstance(data["edges"], list):
                            for edge in data["edges"]:
                                e_type = edge.get("type", "unknown_edge_type")
                                total_edge_types[e_type] += 1
                                
                        total_files_processed += 1

                except json.JSONDecodeError:
                    print(f"Skipping {filename}: Invalid JSON.")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    # --- 2. PRINT TEXT RESULTS ---
    print("\n" + "="*40)
    print(f"ANALYSIS COMPLETE ({total_files_processed} files processed)")
    print("="*40)

    print("\n--- Total Nodes by Type ---")
    if total_node_types:
        for n_type, count in total_node_types.most_common():
            label = n_type if n_type else "<empty>"
            print(f"{label:<25}: {count}")
    else:
        print("No nodes found.")

    print("\n--- Total Edges by Type ---")
    if total_edge_types:
        for e_type, count in total_edge_types.most_common():
            label = e_type if e_type else "<empty>"
            print(f"{label:<25}: {count}")
    else:
        print("No edges found.")

    # --- 3. PLOT AND SAVE ---
    if not total_node_types and not total_edge_types:
        print("No data to plot.")
        return

    # Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot Nodes (Left Subplot) ---
    if total_node_types:
        # Sort and unpack
        sorted_nodes = total_node_types.most_common()
        n_labels, n_counts = zip(*sorted_nodes)
        
        # Rename empty strings for visibility on plot
        n_labels = [lbl if lbl else "<empty>" for lbl in n_labels]

        bars1 = ax1.bar(n_labels, n_counts, color='skyblue', edgecolor='black')
        ax1.set_title('Node Types Distribution', fontsize=14)
        ax1.set_xlabel('Node Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                     ha='center', va='bottom', fontsize=9)

    # --- Plot Edges (Right Subplot) ---
    if total_edge_types:
        sorted_edges = total_edge_types.most_common()
        e_labels, e_counts = zip(*sorted_edges)
        
        e_labels = [lbl if lbl else "<empty>" for lbl in e_labels]

        bars2 = ax2.bar(e_labels, e_counts, color='lightgreen', edgecolor='black')
        ax2.set_title('Edge Types Distribution', fontsize=14)
        ax2.set_xlabel('Edge Type')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    
    print(f"\nSaving plot to '{output_file}'...")
    plt.savefig(output_file, dpi=300)
    print("Done.")

# --- USAGE ---
folders_to_scan = [
    "/home/adamov/fergus/GRID/dataset/GRID_Dataset-master/Mini_Dataset/data/scene.1.graphs",
    "/home/adamov/fergus/GRID/dataset/GRID_Dataset-master/Mini_Dataset/data/scene.2.graphs"
]

analyze_scenegraph_stats(folders_to_scan)