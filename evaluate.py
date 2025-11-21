import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from model import SpectrumTokenizer
from data import SpectrumDataset, collate_fn
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime
import random

# Try importing umap, if not available, fallback to PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not found, falling back to PCA for visualization.")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Universal Spectrum Tokenizer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint to evaluate")
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face dataset cache directory")
    parser.add_argument("--save_dir", type=str, default="./evaluation_results", help="Directory to save plots")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate():
    args = parse_args()
    set_seed(args.seed)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Evaluation Run ID: {run_id}")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Datasets (Must match training logic for consistent splits)
    print("Loading SDSS...")
    sdss_ds = SpectrumDataset("MultimodalUniverse/sdss", cache_dir=args.cache_dir, max_length=args.max_samples)
    print("Loading DESI...")
    desi_ds = SpectrumDataset("MultimodalUniverse/desi", cache_dir=args.cache_dir, max_length=args.max_samples)
    
    full_ds = ConcatDataset([sdss_ds, desi_ds])
    
    # Split Train/Val/Test (80/10/10)
    total_size = len(full_ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(args.seed)
    _, _, test_ds = torch.utils.data.random_split(full_ds, [train_size, val_size, test_size], generator=generator)
    print(f"Dataset split: Test={len(test_ds)}")
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = SpectrumTokenizer(
        patch_size=32,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    # Load Checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    model.eval()
    
    # 1. Reconstructions
    print("Generating reconstruction plots...")
    with torch.no_grad():
        sdss_plotted = 0
        desi_plotted = 0
        total_plotted = 0
        
        for batch in test_loader:
            flux = batch['flux'].to(device)
            wavelength = batch['wavelength'].to(device)
            ivar = batch['ivar'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            obj_ids = batch['object_id']
            dataset_names = batch['dataset_name']
            
            reconstruction = model(flux, wavelength)
            
            for idx, obj_id in enumerate(obj_ids):
                ds_name = dataset_names[idx]
                
                # Logic to ensure we get both if possible
                should_plot = False
                if "sdss" in ds_name.lower() and sdss_plotted < 2:
                    should_plot = True
                    sdss_plotted += 1
                elif "desi" in ds_name.lower() and desi_plotted < 2:
                    should_plot = True
                    desi_plotted += 1
                
                if should_plot:
                    f = flux[idx].cpu().numpy()
                    w = wavelength[idx].cpu().numpy()
                    r = reconstruction[idx].cpu().numpy()
                    v = valid_mask[idx].cpu().numpy().astype(bool)
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(w[v], f[v], label="Original", alpha=0.7)
                    plt.plot(w[v], r[v], label="Reconstruction", alpha=0.7)
                    plt.legend()
                    plt.title(f"Reconstruction {obj_id} ({ds_name})")
                    plt.xlabel("Wavelength")
                    plt.ylabel("Normalized Flux")
                    plt.savefig(os.path.join(args.save_dir, f"eval_reconstruction_{run_id}_{total_plotted}_{ds_name}.png"))
                    plt.close()
                    total_plotted += 1
                
                if total_plotted >= 4:
                    break
            
            if total_plotted >= 4:
                break 

    # 2. UMAP
    print("Extracting embeddings for UMAP...")
    embeddings = []
    redshifts = []
    ds_names_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(device)
            wavelength = batch['wavelength'].to(device)
            z = batch['z']
            ds_names = batch['dataset_name']
            
            emb = model.encode(flux, wavelength) # [B, N, D]
            emb_mean = emb.mean(dim=1) # [B, D]
            
            embeddings.append(emb_mean.cpu().numpy())
            redshifts.append(z.numpy())
            ds_names_list.extend(ds_names)
                
    embeddings = np.concatenate(embeddings, axis=0)
    redshifts = np.concatenate(redshifts, axis=0)
    ds_names_list = np.array(ds_names_list)
    
    print(f"Running dimensionality reduction on {embeddings.shape}...")
    
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.5)
        proj = reducer.fit_transform(embeddings)
        title = "UMAP"
    else:
        reducer = PCA(n_components=2)
        proj = reducer.fit_transform(embeddings)
        title = "PCA"
        
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    
    # Determine global min/max for redshift colorbar
    z_min, z_max = redshifts.min(), redshifts.max()
    
    # Plot SDSS (Left Panel)
    mask_sdss = np.char.find(np.char.lower(ds_names_list.astype(str)), "sdss") != -1
    sc = None
    if mask_sdss.any():
        sc = axes[0].scatter(proj[mask_sdss, 0], proj[mask_sdss, 1], c=redshifts[mask_sdss], 
                              cmap='viridis', s=5, alpha=0.6, marker='o', vmin=z_min, vmax=z_max)
        axes[0].set_title("SDSS")
    else:
        axes[0].text(0.5, 0.5, "No SDSS data", ha='center', va='center')
        axes[0].set_title("SDSS (Empty)")
        
    axes[0].set_xlabel("Dim 1")
    axes[0].set_ylabel("Dim 2")

    # Plot DESI (Right Panel)
    mask_desi = np.char.find(np.char.lower(ds_names_list.astype(str)), "desi") != -1
    if mask_desi.any():
        sc = axes[1].scatter(proj[mask_desi, 0], proj[mask_desi, 1], c=redshifts[mask_desi], 
                              cmap='viridis', s=5, alpha=0.6, marker='^', vmin=z_min, vmax=z_max)
        axes[1].set_title("DESI")
    else:
        axes[1].text(0.5, 0.5, "No DESI data", ha='center', va='center')
        axes[1].set_title("DESI (Empty)")
        
    axes[1].set_xlabel("Dim 1")
    
    # Add colorbar
    if sc:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), label='Redshift')
    
    plt.suptitle(f"{title} of Spectral Embeddings (Test Set)")
    plt.savefig(os.path.join(args.save_dir, f"eval_embedding_projection_{run_id}.png"))
    plt.close()
    print(f"Saved eval_embedding_projection_{run_id}.png")

if __name__ == "__main__":
    evaluate()
