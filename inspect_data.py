
from datasets import load_dataset

def inspect_dataset(dataset_name):
    print(f"Loading {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, streaming=True, split="train")
        print(f"Successfully loaded {dataset_name}")
        item = next(iter(ds))
        print(f"Keys: {item.keys()}")
        if 'spectrum' in item:
            print(f"Spectrum keys: {item['spectrum'].keys()}")
            for k, v in item['spectrum'].items():
                if hasattr(v, 'shape'):
                    print(f"Spectrum {k}: {v.shape}")
                elif isinstance(v, list):
                    print(f"Spectrum {k}: list of len {len(v)}")
                else:
                    print(f"Spectrum {k}: {type(v)}")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")

if __name__ == "__main__":
    inspect_dataset("MultimodalUniverse/sdss")
    inspect_dataset("MultimodalUniverse/desi")
