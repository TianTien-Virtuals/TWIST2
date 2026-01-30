import os
import yaml


def generate_config_yaml(folder_path, output_yaml_path, weight=1.0, description="general movement"):
    """Generate YAML from a single folder with a given weight."""
    config = {
        "root_path": folder_path,
        "motions": []
    }
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".pkl"):
            config["motions"].append({
                "file": file_name,
                "weight": weight,
                "description": description
            })
    with open(output_yaml_path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)
    return config


def collect_pkl_files(folder_abs, folder_rel, recursive):
    """
    Collect .pkl paths under folder_abs, relative to folder_rel's parent (base).
    Returns list of (relative_path, ) for each .pkl.
    """
    collected = []
    if recursive:
        for root, _dirs, files in os.walk(folder_abs):
            for file_name in sorted(files):
                if file_name.endswith(".pkl"):
                    full = os.path.join(root, file_name)
                    # Path relative to base_path: base_path is folder_abs's parent's parent when folder_rel is "a/b"
                    rel = os.path.relpath(full, folder_abs)
                    rel = os.path.join(folder_rel, rel).replace("\\", "/")
                    collected.append(rel)
    else:
        for file_name in sorted(os.listdir(folder_abs)):
            if file_name.endswith(".pkl"):
                collected.append(os.path.join(folder_rel, file_name).replace("\\", "/"))
    return collected


def generate_config_yaml_multi(base_path, folder_weights, output_yaml_path, description="general movement"):
    """
    Generate YAML from multiple folders with different weights.
    base_path: root directory (e.g. path to assets/)
    folder_weights: list of (folder_relative_to_base, weight) or (folder_relative_to_base, weight, recursive).
                    If recursive is True, all .pkl under that folder (including subfolders) are included.
    output_yaml_path: where to write the YAML
    """
    base_path = os.path.abspath(base_path)
    config = {
        "root_path": base_path,
        "motions": []
    }
    for entry in folder_weights:
        if len(entry) == 3:
            folder_rel, weight, recursive = entry
        else:
            folder_rel, weight = entry
            recursive = False
        folder_abs = os.path.join(base_path, folder_rel)
        if not os.path.isdir(folder_abs):
            print(f"Warning: folder does not exist: {folder_abs}")
            continue
        paths = collect_pkl_files(folder_abs, folder_rel, recursive)
        for file_rel in paths:
            config["motions"].append({
                "file": file_rel,
                "weight": float(weight),
                "description": description
            })
    with open(output_yaml_path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)
    print(f"Added {len(config['motions'])} motions from {len(folder_weights)} folder(s)")
    return config


# Script directory: legged_gym/legged_gym/scripts -> repo root is ../../..
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.normpath(os.path.join(LEGGED_GYM_ROOT, ".."))
ASSETS_PATH = os.path.join(REPO_ROOT, "assets")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate motion dataset YAML from assets folders.")
    parser.add_argument("--base", default=ASSETS_PATH, help="Base path (default: repo assets/)")
    parser.add_argument("--output", default=None, help="Output YAML path (default: motion_data_configs/twist2_dataset.yaml)")
    parser.add_argument("--list", action="store_true", help="Only list folders and exit")
    args = parser.parse_args()
    base_path = os.path.abspath(args.base)
    output_yaml_path = args.output or os.path.join(LEGGED_GYM_ROOT, "motion_data_configs", "twist2_residual_dataset.yaml")
    # TWIST2_data (weight 1.0, recursive); eastworld PICO data; example_motions (weight 8.0, flat)
    folder_weights = [
        # ("TWIST2_full/TWIST2_data", 1.0, True),   # recursive: include all .pkl in subfolders
        ("TWIST2_full/eastworld_data/eastworlds_tt", 1.0, True),  # PICO recorded .pkl (add weight as needed)
        # ("example_motions", 8.0),                 # flat: only direct .pkl
    ]
    if args.list:
        for entry in folder_weights:
            rel = entry[0]
            w = entry[1]
            rec = entry[2] if len(entry) == 3 else False
            p = os.path.join(base_path, rel)
            if not os.path.isdir(p):
                print(f"  {rel} (weight {w}, recursive={rec}) -> not found")
                continue
            n = len(collect_pkl_files(p, rel, rec))
            print(f"  {rel} (weight {w}, recursive={rec}) -> {n} .pkl files")
        print(f"Base: {base_path}")
        exit(0)
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    generate_config_yaml_multi(base_path, folder_weights, output_yaml_path)
    print(f"YAML configuration written to: {output_yaml_path}")
