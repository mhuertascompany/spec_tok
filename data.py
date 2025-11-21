
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

class SpectrumDataset(Dataset):
    def __init__(self, dataset_name, split="train", max_length=None, cache_dir=None):
        if max_length is not None:
            # Use slicing syntax for non-streaming dataset
            split = f"{split}[:{max_length}]"
            
        print(f"Loading {dataset_name} with split={split} (streaming=False)...")
        # Use streaming=False to leverage Arrow memory mapping and avoid RAM issues
        self.ds = load_dataset(dataset_name, split=split, streaming=False, cache_dir=cache_dir)
        print(f"Loaded {len(self.ds)} examples.")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        spec = item['spectrum']
        
        flux = np.array(spec['flux'], dtype=np.float32)
        ivar = np.array(spec['ivar'], dtype=np.float32)
        wavelength = np.array(spec['lambda'], dtype=np.float32)
        mask = np.array(spec['mask'], dtype=np.float32)
        
        # Metadata
        z = float(item.get('Z', -1.0))
        obj_id = str(item.get('object_id', ''))
        
        # Normalize by median flux
        valid_mask = ivar > 0
        if valid_mask.sum() > 0:
            median_flux = np.median(flux[valid_mask])
            if median_flux == 0 or np.isnan(median_flux):
                median_flux = 1.0
        else:
            median_flux = 1.0
            
        flux = flux / median_flux
        # Normalized ivar = ivar * median_flux^2
        ivar = ivar * (median_flux**2)
        
        return {
            "flux": torch.tensor(flux),
            "ivar": torch.tensor(ivar),
            "wavelength": torch.tensor(wavelength),
            "mask": torch.tensor(mask),
            "valid_mask": torch.tensor(valid_mask),
            "z": torch.tensor(z, dtype=torch.float32),
            "object_id": obj_id,
            "dataset_name": "SDSS" if "sdss" in str(item.get('object_id', '')).lower() else "DESI" # Heuristic or pass in
        }

def collate_fn(batch):
    # Pad to max length in batch
    max_len = max([b['flux'].shape[0] for b in batch])
    
    flux_batch = []
    ivar_batch = []
    wavelength_batch = []
    mask_batch = []
    valid_mask_batch = []
    z_batch = []
    id_batch = []
    
    for b in batch:
        l = b['flux'].shape[0]
        pad_len = max_len - l
        
        # Pad flux with 0
        f = torch.nn.functional.pad(b['flux'], (0, pad_len), value=0)
        flux_batch.append(f)
        
        # Pad ivar with 0 (invalid)
        i = torch.nn.functional.pad(b['ivar'], (0, pad_len), value=0)
        ivar_batch.append(i)
        
        # Pad wavelength with 0
        w = torch.nn.functional.pad(b['wavelength'], (0, pad_len), value=0)
        wavelength_batch.append(w)
        
        # Pad mask
        m = torch.nn.functional.pad(b['mask'], (0, pad_len), value=0)
        mask_batch.append(m)
        
        # Pad valid_mask with False
        vm = torch.nn.functional.pad(b['valid_mask'], (0, pad_len), value=False)
        valid_mask_batch.append(vm)
        
        z_batch.append(b['z'])
        id_batch.append(b['object_id'])
        
    return {
        "flux": torch.stack(flux_batch),
        "ivar": torch.stack(ivar_batch),
        "wavelength": torch.stack(wavelength_batch),
        "mask": torch.stack(mask_batch),
        "valid_mask": torch.stack(valid_mask_batch),
        "z": torch.stack(z_batch),
        "object_id": id_batch,
        "dataset_name": [b['dataset_name'] for b in batch]
    }
