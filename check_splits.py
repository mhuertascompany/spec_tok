from datasets import get_dataset_split_names

try:
    print("SDSS splits:", get_dataset_split_names("MultimodalUniverse/sdss"))
    print("DESI splits:", get_dataset_split_names("MultimodalUniverse/desi"))
except Exception as e:
    print(f"Error: {e}")
