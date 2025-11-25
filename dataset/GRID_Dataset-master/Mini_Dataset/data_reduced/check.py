import json
from collections import Counter

# Example: load one scene graph
with open("scene.2.scene_graph.json", "r") as f:
    sg = json.load(f)

nodes = sg["nodes"]

# Extract labels (you can decide whether to include empty labels or not)
labels = []
for n in nodes:
    label = n["attributes"].get("label", "")
    if label:  # skip empty labels; remove this `if` if you want to count them
        labels.append(label)

label_counts = Counter(labels)

print("All labels:", labels)
print("Duplicate labels (label -> count):")
for lab, c in label_counts.items():
    if c > 1:
        print(f"  {lab}: {c} times")

if all(c == 1 for c in label_counts.values()):
    print("✅ All non-empty labels are unique in this scene graph.")