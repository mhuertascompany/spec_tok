from datasets import load_dataset

try:
    print("Attempting to load SDSS with streaming=False and split slicing...")
    ds = load_dataset("MultimodalUniverse/sdss", split="train[:10]", streaming=False)
    print("Success!")
    print(ds)
    print(ds[0].keys())
except Exception as e:
    print(f"Failed: {e}")
