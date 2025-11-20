
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from model import SpectrumTokenizer
from data import SpectrumDataset, collate_fn
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from sklearn.decomposition import PCA

# Try importing umap, if not available, fallback to PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not found, falling back to PCA for visualization.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Universal Spectrum Tokenizer")
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face dataset cache directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints and plots")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset")
    return parser.parse_args()

def train():
    args = parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Datasets
    print("Loading SDSS...")
    sdss_ds = SpectrumDataset("MultimodalUniverse/sdss", cache_dir=args.cache_dir, max_length=args.max_samples)
    print("Loading DESI...")
    desi_ds = SpectrumDataset("MultimodalUniverse/desi", cache_dir=args.cache_dir, max_length=args.max_samples)
    
    train_ds = ConcatDataset([sdss_ds, desi_ds])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Model
    model = SpectrumTokenizer(
        patch_size=32,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    print("Starting training...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0
        
        for i, batch in enumerate(train_loader):
            flux = batch['flux'].to(device)
            wavelength = batch['wavelength'].to(device)
            ivar = batch['ivar'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            reconstruction = model(flux, wavelength)
            
            # Loss: Weighted MSE (Gaussian Likelihood)
            # L = sum(ivar * (y - y_hat)^2) / sum(ivar)
            # Only on valid pixels
            
            diff = (flux - reconstruction) ** 2
            weighted_diff = diff * ivar
            
            # Mask out invalid pixels (ivar should be 0 there anyway, but valid_mask ensures it)
            loss = (weighted_diff * valid_mask).sum() / (ivar.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}")
                
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    # Evaluation and Plotting
    print("Generating diagnostic plots...")
    model.eval()
    
    # 1. Reconstructions
    # We want to plot a few examples from SDSS and DESI
    # We can iterate through the dataset to find them
    
    sdss_example = None
    desi_example = None
    
    # Just grab a batch and check IDs
    with torch.no_grad():
        for batch in train_loader:
            flux = batch['flux'].to(device)
            wavelength = batch['wavelength'].to(device)
            ivar = batch['ivar'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            obj_ids = batch['object_id']
            
            reconstruction = model(flux, wavelength)
            
            for idx, obj_id in enumerate(obj_ids):
                # Heuristic to identify SDSS vs DESI if not explicit
                # But we can also just check which dataset they came from if we tracked it.
                # In data.py I added 'dataset_name' to __getitem__ but collate_fn didn't stack it (it's string).
                # But I didn't add it to collate_fn return.
                # Let's just rely on the fact that we loaded them.
                # Or check the ID format if possible.
                # Actually, let's just plot the first few and label them by ID.
                
                # Better: Let's try to find one of each.
                # SDSS IDs are usually long integers. DESI IDs are also long integers.
                # Let's just plot 4 random ones.
                
                if idx < 4:
                    f = flux[idx].cpu().numpy()
                    w = wavelength[idx].cpu().numpy()
                    r = reconstruction[idx].cpu().numpy()
                    v = valid_mask[idx].cpu().numpy().astype(bool)
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(w[v], f[v], label="Original", alpha=0.7)
                    plt.plot(w[v], r[v], label="Reconstruction", alpha=0.7)
                    plt.legend()
                    plt.title(f"Reconstruction {obj_id}")
                    plt.xlabel("Wavelength")
                    plt.ylabel("Normalized Flux")
                    plt.savefig(os.path.join(args.save_dir, f"reconstruction_{epoch}_{idx}.png"))
                    plt.close()
            
            break # Just one batch for reconstruction plots

    # 2. UMAP
    # We need to extract embeddings for a subset of data
    print("Extracting embeddings for UMAP...")
    embeddings = []
    redshifts = []
    
    # Use a subset (e.g. 1000 samples)
    count = 0
    max_samples = 1000
    
    with torch.no_grad():
        for batch in train_loader:
            flux = batch['flux'].to(device)
            wavelength = batch['wavelength'].to(device)
            z = batch['z']
            
            # We need to get the embedding from the model.
            # model.forward returns reconstruction.
            # We need to modify model to return embeddings or add a method.
            # Let's add a method `encode` to SpectrumTokenizer in model.py?
            # Or just access `model.encoder` here if we replicate the forward pass logic.
            # Replicating logic is safer than modifying model.py if we don't want to break things,
            # but modifying model.py is cleaner.
            # Let's modify model.py to return embeddings if requested, or add encode method.
            # For now, I'll do it here by copy-pasting logic (hacky but fast) or just modify model.py.
            # I will modify model.py in the next step.
            # Assuming model.encode(flux, wavelength) exists.
            
            # Wait, I can't modify model.py in this `write_to_file` call.
            # I will assume `model.encode` exists and implement it in the next step.
            
            emb = model.encode(flux, wavelength) # [B, N, D]
            
            # Mean pool
            emb_mean = emb.mean(dim=1) # [B, D]
            
            embeddings.append(emb_mean.cpu().numpy())
            redshifts.append(z.numpy())
            
            count += flux.shape[0]
            if count >= max_samples:
                break
                
    embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
    redshifts = np.concatenate(redshifts, axis=0)[:max_samples]
    
    print(f"Running dimensionality reduction on {embeddings.shape}...")
    
    if HAS_UMAP:
        reducer = umap.UMAP()
        proj = reducer.fit_transform(embeddings)
        title = "UMAP"
    else:
        reducer = PCA(n_components=2)
        proj = reducer.fit_transform(embeddings)
        title = "PCA"
        
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=redshifts, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(sc, label='Redshift')
    plt.title(f"{title} of Spectral Embeddings")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig(os.path.join(args.save_dir, "embedding_projection.png"))
    plt.close()
    print("Saved embedding_projection.png")

if __name__ == "__main__":
    train()
