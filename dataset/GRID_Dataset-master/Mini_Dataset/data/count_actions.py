import json
from collections import Counter
import os
import matplotlib.pyplot as plt

def count_and_plot_actions(file_paths, output_file="action_distribution.png"):
    # Initialize a single counter to aggregate stats across all files
    total_action_counts = Counter()
    total_commands_processed = 0

    # --- 1. DATA COLLECTION ---
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File '{file_path}' not found. Skipping.")
            continue

        print(f"Scanning file: {file_path} ...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "commands" in data and isinstance(data["commands"], list):
                for item in data["commands"]:
                    total_commands_processed += 1
                    
                    if "low" in item and isinstance(item["low"], list):
                        for action_string in item["low"]:
                            if isinstance(action_string, str):
                                parts = action_string.strip().split()
                                if parts:
                                    action_type = parts[0]
                                    total_action_counts[action_type] += 1
            else:
                print(f"Warning: 'commands' key not found or invalid in {file_path}")

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{file_path}'. Skipping.")
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")

    # --- 2. PRINT TEXT RESULTS ---
    print("\n" + "="*40)
    print(f"ANALYSIS COMPLETE")
    print(f"Files Scanned: {len(file_paths)}")
    print(f"Total Commands Processed: {total_commands_processed}")
    print("="*40)
    
    if not total_action_counts:
        print("No actions found to plot.")
        return

    print("\n--- Total Actions by Type ---")
    sorted_actions = total_action_counts.most_common()
    
    for action, count in sorted_actions:
        print(f"{action:<20}: {count}")

    # --- 3. PLOT AND SAVE ---
    actions, counts = zip(*sorted_actions)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(actions, counts, color='skyblue', edgecolor='black')
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.title('Distribution of Action Types', fontsize=16)
    plt.xlabel('Action Type', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # SAVE instead of SHOW
    print(f"\nSaving plot to '{output_file}'...")
    plt.savefig(output_file, dpi=300)
    print("Done.")

# --- USAGE ---
files_to_scan = [
    "/home/adamov/fergus/GRID/dataset/GRID_Dataset-master/Mini_Dataset/data/scene.1.instr.json",
    "/home/adamov/fergus/GRID/dataset/GRID_Dataset-master/Mini_Dataset/data/scene.2.instr.json"
]

count_and_plot_actions(files_to_scan)