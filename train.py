
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
    parser = argparse.ArgumentParser(description="Train Universal Spectrum Tokenizer")
    parser.add_argument("--cache_dir", type=str, default=None, help="Hugging Face dataset cache directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints and plots")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    args = parse_args()
    set_seed(args.seed)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Datasets
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
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, val_size, test_size], generator=generator)
    print(f"Dataset split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = SpectrumTokenizer(
        patch_size=32,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'loss' in checkpoint: # Assuming 'loss' in checkpoint refers to best_val_loss
                best_val_loss = checkpoint['loss']
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {args.resume_from}, starting from scratch.")
    
    print("Starting training...")
    
    patience_counter = 0
    best_model_path = ""
    
    for epoch in range(start_epoch, args.epochs):
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for i, batch in enumerate(train_loader):
            flux = batch['flux'].to(device)
            wavelength = batch['wavelength'].to(device)
            ivar = batch['ivar'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            optimizer.zero_grad()
            
            reconstruction = model(flux, wavelength)
            
            diff = (flux - reconstruction) ** 2
            weighted_diff = diff * ivar
            loss = (weighted_diff * valid_mask).sum() / (ivar.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
            
        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                flux = batch['flux'].to(device)
                wavelength = batch['wavelength'].to(device)
                ivar = batch['ivar'].to(device)
                valid_mask = batch['valid_mask'].to(device)
                
                reconstruction = model(flux, wavelength)
                
                diff = (flux - reconstruction) ** 2
                weighted_diff = diff * ivar
                loss = (weighted_diff * valid_mask).sum() / (ivar.sum() + 1e-8)
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_path = os.path.join(args.save_dir, f"best_model_{run_id}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break
            
        # Save regular checkpoint
        ckpt_path = os.path.join(args.save_dir, f"model_{run_id}_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, ckpt_path)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(args.save_dir, f"loss_curve_{run_id}.png"))
    plt.close()
    print(f"Saved loss_curve_{run_id}.png")

    # Evaluation and Plotting on Test Set
    print("Generating diagnostic plots on Test Set...")
    # Load best model
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found, using current model.")
        
    model.eval()
    
    # 1. Reconstructions
    # 1. Reconstructions
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
                    plt.savefig(os.path.join(args.save_dir, f"reconstruction_{run_id}_test_{total_plotted}_{ds_name}.png"))
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
        
    plt.figure(figsize=(10, 8))
    
    # Plot SDSS
    mask_sdss = np.char.find(np.char.lower(ds_names_list.astype(str)), "sdss") != -1
    if mask_sdss.any():
        sc1 = plt.scatter(proj[mask_sdss, 0], proj[mask_sdss, 1], c=redshifts[mask_sdss], 
                          cmap='viridis', s=5, alpha=0.6, marker='o', label='SDSS', vmin=redshifts.min(), vmax=redshifts.max())
    
    # Plot DESI
    mask_desi = np.char.find(np.char.lower(ds_names_list.astype(str)), "desi") != -1
    if mask_desi.any():
        sc2 = plt.scatter(proj[mask_desi, 0], proj[mask_desi, 1], c=redshifts[mask_desi], 
                          cmap='viridis', s=5, alpha=0.6, marker='^', label='DESI', vmin=redshifts.min(), vmax=redshifts.max())
    
    plt.colorbar(sc1 if mask_sdss.any() else sc2, label='Redshift')
    plt.legend()
    plt.title(f"{title} of Spectral Embeddings (Test Set)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig(os.path.join(args.save_dir, f"embedding_projection_{run_id}.png"))
    plt.close()
    print(f"Saved embedding_projection_{run_id}.png")

if __name__ == "__main__":
    train()
