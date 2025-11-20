
import torch
import torch.nn as nn
import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, min_period, max_period):
        super().__init__()
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        
        # Create frequencies
        # The paper says: omega_k in [2pi/max_period, 2pi/min_period]
        # log-spaced
        half_dim = dim // 2
        
        # We want half_dim frequencies.
        # start: 2*pi/max_period
        # end: 2*pi/min_period
        # In log space: log(start) to log(end)
        
        start = math.log(2 * math.pi / max_period)
        end = math.log(2 * math.pi / min_period)
        
        frequencies = torch.exp(torch.linspace(start, end, half_dim))
        self.register_buffer('frequencies', frequencies)

    def forward(self, x):
        # x: [batch, seq_len] (wavelengths)
        # output: [batch, seq_len, dim]
        
        # args: [batch, seq_len, half_dim]
        args = x.unsqueeze(-1) * self.frequencies.unsqueeze(0).unsqueeze(0)
        
        emb_sin = torch.sin(args)
        emb_cos = torch.cos(args)
        
        # Concatenate sin and cos
        # [batch, seq_len, dim]
        emb = torch.cat([emb_sin, emb_cos], dim=-1)
        
        return emb

class SpectrumTokenizer(nn.Module):
    def __init__(
        self,
        patch_size=32,
        embed_dim=512,
        depth=6,
        num_heads=8,
        min_period=1.0, # Angstroms? Need to check typical values.
        max_period=10000.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Input projection: Patch size -> Embed dim
        # Input is flux (1 channel)
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        
        # Wavelength embedding
        self.wavelength_embed = SinusoidalEmbedding(embed_dim, min_period, max_period)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            batch_first=True,
            norm_first=True # Pre-norm is standard for ViT
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, patch_size)
        
    def patchify(self, x):
        # x: [batch, seq_len]
        # Returns: [batch, num_patches, patch_size]
        # Pad if needed
        B, L = x.shape
        pad_len = (self.patch_size - (L % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, pad_len))
        
        x = x.view(B, -1, self.patch_size)
        return x, L

    def unpatchify(self, x, original_length):
        # x: [batch, num_patches, patch_size]
        B = x.shape[0]
        x = x.view(B, -1)
        return x[:, :original_length]

    def forward(self, flux, wavelength, mask=None):
        # flux: [batch, seq_len]
        # wavelength: [batch, seq_len]
        # mask: [batch, seq_len] (1 for valid, 0 for invalid) - Optional for now
        
        # 1. Patchify Flux
        flux_patches, original_len = self.patchify(flux) # [B, N, P]
        
        # 2. Patchify Wavelength (take mean or center? Paper says "Wavelength embeddings are patched")
        # "Wavelength embeddings are patched and added to the flux patches"
        # This implies we calculate embedding for each pixel, then patchify? 
        # Or patchify wavelength then embed?
        # "For wavelength lambda ... the embedding is ... Wavelength embeddings are patched"
        # It likely means we take the embedding of the wavelength grid, and then maybe average it per patch?
        # Or maybe the embedding is per patch?
        # Let's re-read: "Wavelength embeddings are patched and added to the flux patches"
        # If we embed every pixel, we get [B, L, D]. Patching gives [B, N, P, D].
        # Flux patches are [B, N, P]. Projected to [B, N, D].
        # If we add them, they must match.
        # Maybe we project flux [B, N, P] -> [B, N, D].
        # And we need wavelength embedding [B, N, D].
        # If we embed every pixel, we have [B, L, D]. Patchify -> [B, N, P, D].
        # This doesn't match [B, N, D] unless we flatten P*D? No.
        # Usually in ViT, position embedding is added to the patch embedding.
        # If the position embedding is pixel-wise, how do we add it to a patch token?
        # Maybe we average the wavelength embeddings within a patch?
        # Or maybe we embed the central wavelength of the patch?
        # Paper: "Wavelength embeddings are patched and added to the flux patches"
        # This is slightly ambiguous. 
        # Option A: Project flux patch [P] -> [D]. Embed all wavelengths in patch [P] -> [P, D_emb]. 
        # This doesn't sum easily.
        # Option B: Project flux patch [P] -> [D]. Embed central wavelength -> [D]. Add.
        # Option C: The "patch projection" is actually a linear layer on the flattened patch.
        # If we have pixel-wise embeddings, maybe we process them similarly?
        # Let's assume we take the mean wavelength of the patch and embed that.
        # Or we embed all pixels and then pass through a linear layer?
        
        # Let's look at the paper again.
        # "Wavelength embeddings are patched and added to the flux patches"
        # Maybe it means:
        # Flux: [B, N, P] -> Linear -> [B, N, D]
        # Wavelength: [B, L] -> Embed -> [B, L, D] -> Patchify -> [B, N, P, D] -> Linear? -> [B, N, D]?
        # Or maybe just Average pooling?
        
        # Let's try: Embed(Mean(Wavelength_patch))
        
        wavelength_patches, _ = self.patchify(wavelength) # [B, N, P]
        patch_wavelengths = wavelength_patches.mean(dim=-1) # [B, N]
        
        pos_embed = self.wavelength_embed(patch_wavelengths) # [B, N, D]
        
        # Flux embedding
        x = self.patch_embed(flux_patches) # [B, N, D]
        
        # Add pos embed
        x = x + pos_embed
        
        # Encoder
        memory = self.encoder(x) # [B, N, D]
        
        # Decoder
        # For reconstruction, we query at the same wavelengths.
        # So decoder input is just the target position embeddings?
        # "The decoder mirrors the encoder; the decoding process proceeds by first requesting an output wavelength grid, and adding its sinusoidal embedding to the encoder tokens"
        # Wait, "adding its sinusoidal embedding to the encoder tokens"??
        # Usually decoder takes "query" tokens.
        # If we are reconstructing the SAME grid, we can use the same pos_embed.
        # But if we want to reconstruct on a NEW grid, we need new pos_embed.
        # The paper says: "adding its sinusoidal embedding to the encoder tokens"
        # This sounds like the decoder input is (EncoderOutput + TargetPosEmbed).
        # But EncoderOutput is [B, N, D]. TargetPosEmbed is [B, N_target, D].
        # If N != N_target, we can't add.
        # Maybe it means Cross-Attention?
        # "The decoder then processes the tokens to produce a sequence of outputs"
        # Standard Transformer Decoder:
        # Input: Target Sequence (shifted).
        # Memory: Encoder Output.
        # Here we are doing something like Masked Autoencoder or just Autoencoder.
        # If it's a pure Autoencoder, we might just pass EncoderOutput to Decoder?
        # But we want to query at specific wavelengths.
        # This suggests the Decoder Input is the Target Pos Embeddings.
        # And it cross-attends to the Encoder Output.
        
        # Let's assume:
        # Decoder Input = Target Pos Embeddings.
        # Cross Attention to Encoder Output.
        
        target_pos_embed = pos_embed # For autoencoding same grid
        
        # We need a "start token" or just feed the pos embeddings?
        # In MAE, decoder inputs are (MaskedTokens + PosEmbed).
        # Here we don't have masked tokens in the same way?
        # "adding its sinusoidal embedding to the encoder tokens"
        # This phrasing is confusing.
        # If it means "Decoder takes Encoder tokens AND adds target pos embedding", that implies N=N_target.
        # But the point is to handle arbitrary grids.
        # So it MUST be Cross-Attention.
        # Decoder Input: Target Pos Embeddings (as Query).
        # Encoder Output: Key/Value.
        
        # Let's implement Decoder taking Target Pos Embed as input.
        
        # TransformerDecoder takes:
        # tgt: [B, N_target, D]
        # memory: [B, N_source, D]
        
        # So tgt = target_pos_embed.
        
        dec_out = self.decoder(tgt=target_pos_embed, memory=memory) # [B, N, D]
        
        # Output projection
        out_patches = self.output_proj(dec_out) # [B, N, P]
        
        # Unpatchify
        reconstruction = self.unpatchify(out_patches, original_len)
        
        return reconstruction

    def encode(self, flux, wavelength):
        # flux: [batch, seq_len]
        # wavelength: [batch, seq_len]
        
        flux_patches, _ = self.patchify(flux) # [B, N, P]
        wavelength_patches, _ = self.patchify(wavelength) # [B, N, P]
        patch_wavelengths = wavelength_patches.mean(dim=-1) # [B, N]
        
        pos_embed = self.wavelength_embed(patch_wavelengths) # [B, N, D]
        
        x = self.patch_embed(flux_patches) # [B, N, D]
        x = x + pos_embed
        
        memory = self.encoder(x) # [B, N, D]
        return memory

