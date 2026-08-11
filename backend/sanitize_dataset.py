import os
import yaml

def sanitize_labels():
    dataset_dir = "/home/lohit/realcode/ML/OWN/traffic_analysis/plate_dataset"
    subsets = ["train", "valid", "test"]
    
    total_modified = 0
    total_files = 0
    
    for subset in subsets:
        labels_path = os.path.join(dataset_dir, subset, "labels")
        if not os.path.exists(labels_path):
            print(f"Warning: Path not found {labels_path}")
            continue
            
        for file_name in os.listdir(labels_path):
            if not file_name.endswith(".txt"):
                continue
            
            file_path = os.path.join(labels_path, file_name)
            total_files += 1
            
            with open(file_path, "r") as f:
                lines = f.readlines()
                
            modified_lines = []
            modified = False
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = parts[0]
                # If class_id is not '0', change it to '0' (license_plate)
                if class_id != "0":
                    parts[0] = "0"
                    modified = True
                
                modified_lines.append(" ".join(parts) + "\n")
                
            if modified:
                with open(file_path, "w") as f:
                    f.writelines(modified_lines)
                total_modified += 1

    print("Sanitization complete:")
    print(f"  - Total label files processed: {total_files}")
    print(f"  - Label files updated/unified to class 0: {total_modified}")

def update_yaml():
    yaml_path = "/home/lohit/realcode/ML/OWN/traffic_analysis/plate_dataset/data.yaml"
    if not os.path.exists(yaml_path):
        print(f"Error: data.yaml not found at {yaml_path}")
        return
        
    data = {
        "path": "/home/lohit/realcode/ML/OWN/traffic_analysis/plate_dataset",
        "train": "train/images",
        "val": "train/images",
        "test": "train/images",
        "nc": 1,
        "names": ["license_plate"]
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
        
    print(f"Successfully updated {yaml_path} with unified class and absolute paths!")

if __name__ == "__main__":
    sanitize_labels()
    update_yaml()
