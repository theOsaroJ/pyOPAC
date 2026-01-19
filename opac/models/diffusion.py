"""
Diffusion models for molecular property prediction and generation.

This module implements:
1. Diffusion-based property prediction (probabilistic predictions with uncertainty)
2. Diffusion-based molecular generation (alternative to VAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from opac.utils.logger import get_logger

logger = get_logger(__name__)


class DiffusionPropertyPredictor(nn.Module):
    """
    Diffusion model for probabilistic property prediction.
    
    Instead of predicting a single value, this model learns the distribution
    of properties, allowing for uncertainty quantification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Args:
            input_dim: Dimension of input descriptors
            hidden_dim: Hidden layer dimension
            output_dim: Number of target properties
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        super(DiffusionPropertyPredictor, self).__init__()
        
        self.num_timesteps = num_timesteps
        self.output_dim = output_dim
        
        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Denoising network (U-Net style architecture)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Main denoising network
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim * 2),  # input + time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Predict noise at timestep t.
        
        Args:
            x: Noisy property values [batch_size, output_dim]
            t: Timestep [batch_size, 1]
            descriptors: Input descriptors [batch_size, input_dim]
        
        Returns:
            Predicted noise [batch_size, output_dim]
        """
        # Time embedding
        t_emb = self.time_embed(t.float())
        
        # Project descriptors
        desc_emb = self.input_proj(descriptors)
        
        # Concatenate
        combined = torch.cat([x, desc_emb + t_emb], dim=1)
        
        # Predict noise
        return self.network(combined)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add noise to properties according to the forward diffusion process.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Single denoising step (reverse diffusion).
        """
        # Predict noise
        predicted_noise = self.forward(x_t, t, descriptors)
        
        # Compute coefficients
        alpha_t = self.alphas[t].reshape(-1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1)
        beta_t = self.betas[t].reshape(-1, 1)
        
        # Predict x_0
        pred_x_start = (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Compute mean of posterior
        posterior_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise)
        
        # Sample
        if t[0] > 0:
            posterior_variance = beta_t * (1.0 - self.alphas_cumprod_prev[t]) / (1.0 - alpha_cumprod_t)
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            return posterior_mean
    
    @torch.no_grad()
    def p_sample_loop(self, descriptors: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Full reverse diffusion process: sample properties from noise.
        
        Args:
            descriptors: Input descriptors [batch_size, input_dim]
            shape: Shape of output [batch_size, output_dim]
        
        Returns:
            Sampled properties [batch_size, output_dim]
        """
        device = next(self.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, descriptors)
        
        return x_t
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        descriptors: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict properties with uncertainty quantification.
        
        Args:
            descriptors: Input descriptors [batch_size, input_dim]
            num_samples: Number of samples for uncertainty estimation
        
        Returns:
            mean: Mean predictions [batch_size, output_dim]
            std: Standard deviation [batch_size, output_dim]
        """
        self.eval()
        batch_size = descriptors.shape[0]
        output_dim = self.output_dim
        
        # Sample multiple times
        samples = []
        for _ in range(num_samples):
            sample = self.p_sample_loop(descriptors, (batch_size, output_dim))
            samples.append(sample)
        
        samples = torch.stack(samples, dim=0)  # [num_samples, batch_size, output_dim]
        
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        
        return mean, std


class DiffusionMoleculeGenerator(nn.Module):
    """
    Diffusion model for generating molecular descriptors.
    
    This can be used as an alternative to VAE for molecular generation.
    """
    
    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int = 256,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Args:
            descriptor_dim: Dimension of molecular descriptors
            hidden_dim: Hidden layer dimension
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        super(DiffusionMoleculeGenerator, self).__init__()
        
        self.num_timesteps = num_timesteps
        self.descriptor_dim = descriptor_dim
        
        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising network
        self.network = nn.Sequential(
            nn.Linear(descriptor_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, descriptor_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise at timestep t."""
        t_emb = self.time_embed(t.float())
        combined = torch.cat([x, t_emb], dim=1)
        return self.network(combined)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to descriptors."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single denoising step."""
        predicted_noise = self.forward(x_t, t)
        
        alpha_t = self.alphas[t].reshape(-1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1)
        beta_t = self.betas[t].reshape(-1, 1)
        
        posterior_mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - beta_t / torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise
        )
        
        if t[0] > 0:
            posterior_variance = beta_t
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            return posterior_mean
    
    @torch.no_grad()
    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate molecular descriptors from noise.
        
        Args:
            batch_size: Number of molecules to generate
            device: Device to run on
        
        Returns:
            Generated descriptors [batch_size, descriptor_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        x_t = torch.randn((batch_size, self.descriptor_dim), device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t)
        
        return x_t


def train_diffusion_property_predictor(
    dataset,
    input_dim: int,
    output_dim: int,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    hidden_dim: int = 128,
    num_timesteps: int = 1000,
    device: Optional[torch.device] = None,
):
    """
    Train a diffusion model for property prediction.
    
    Args:
        dataset: MoleculeDataset with descriptors and targets
        input_dim: Dimension of input descriptors
        output_dim: Number of target properties
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        num_timesteps: Number of diffusion timesteps
        device: Device to train on
    
    Returns:
        Trained model
    """
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DiffusionPropertyPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_timesteps=num_timesteps
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    # Register buffers on device
    model.betas = model.betas.to(device)
    model.alphas = model.alphas.to(device)
    model.alphas_cumprod = model.alphas_cumprod.to(device)
    model.alphas_cumprod_prev = F.pad(model.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
    model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
    model.sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod.to(device)
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            descriptors = batch['descriptors'].to(device)
            targets = batch['targets'].to(device)
            
            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (descriptors.shape[0],), device=device)
            
            # Sample noise
            noise = torch.randn_like(targets)
            
            # Add noise to targets
            noisy_targets = model.q_sample(targets, t, noise)
            
            # Predict noise
            predicted_noise = model(noisy_targets, t.unsqueeze(1), descriptors)
            
            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def train_diffusion_generator(
    dataset,
    descriptor_dim: int,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    hidden_dim: int = 256,
    num_timesteps: int = 1000,
    device: Optional[torch.device] = None,
):
    """
    Train a diffusion model for molecular generation.
    
    Args:
        dataset: MoleculeDataset with descriptors only
        descriptor_dim: Dimension of descriptors
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Hidden dimension
        num_timesteps: Number of diffusion timesteps
        device: Device to train on
    
    Returns:
        Trained model
    """
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = DiffusionMoleculeGenerator(
        descriptor_dim=descriptor_dim,
        hidden_dim=hidden_dim,
        num_timesteps=num_timesteps
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    # Register buffers on device
    model.betas = model.betas.to(device)
    model.alphas = model.alphas.to(device)
    model.alphas_cumprod = model.alphas_cumprod.to(device)
    model.sqrt_alphas_cumprod = model.sqrt_alphas_cumprod.to(device)
    model.sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod.to(device)
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            descriptors = batch['descriptors'].to(device)
            
            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (descriptors.shape[0],), device=device)
            
            # Sample noise
            noise = torch.randn_like(descriptors)
            
            # Add noise to descriptors
            noisy_descriptors = model.q_sample(descriptors, t, noise)
            
            # Predict noise
            predicted_noise = model(noisy_descriptors, t.unsqueeze(1))
            
            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model
