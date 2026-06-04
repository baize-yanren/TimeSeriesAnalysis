import random
import numpy as np
import torch
from typing import Iterator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    loss_fn: torch.nn.Module,
    real_dataloader: Iterator[torch.Tensor],
    latent_dataloader: Iterator[torch.Tensor],
    discriminator_optimizer: torch.optim.Optimizer,
    generator_optimizer: torch.optim.Optimizer,
    normal_label: int,
    anomaly_label: int,
    epoch: int,
    log_every: int = 10
) -> tuple:
    generator.train()
    discriminator.train()
    
    device = next(generator.parameters()).device
    
    epoch_disc_loss = 0.0
    epoch_gen_loss = 0.0
    num_batches = 0
    
    for batch_idx, real in enumerate(real_dataloader):
        real = real.to(device).float()
        batch_size = real.size(0)

        if batch_idx == 0:
            print(f"[DEBUG] real shape: {real.shape}, device: {real.device}")
            print(f"[DEBUG] Generator device: {next(generator.parameters()).device}")

        real_labels = torch.full((batch_size, 1), normal_label, dtype=torch.float32, device=device)
        fake_labels = torch.full((batch_size, 1), anomaly_label, dtype=torch.float32, device=device)

        z = next(latent_dataloader)
        z = z[:batch_size].to(device).float()

        if batch_idx == 0:
            print(f"[DEBUG] z shape: {z.shape}, device: {z.device}")

        fake = generator(z)
        
        discriminator_optimizer.zero_grad()
        real_output = discriminator(real)
        fake_output = discriminator(fake.detach())
        disc_loss_real = loss_fn(real_output, real_labels)
        disc_loss_fake = loss_fn(fake_output, fake_labels)
        disc_loss = disc_loss_real + disc_loss_fake
        disc_loss.backward()
        discriminator_optimizer.step()
        
        generator_optimizer.zero_grad()
        fake_output = discriminator(fake)
        gen_loss = loss_fn(fake_output, real_labels)
        gen_loss.backward()
        generator_optimizer.step()
        
        epoch_disc_loss += disc_loss.item()
        epoch_gen_loss += gen_loss.item()
        num_batches += 1
        
        if batch_idx % log_every == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(real_dataloader)}, '
                  f'Disc Loss: {disc_loss.item():.6f}, Gen Loss: {gen_loss.item():.6f}')
    
    avg_disc_loss = epoch_disc_loss / num_batches
    avg_gen_loss = epoch_gen_loss / num_batches
    
    return avg_gen_loss, avg_disc_loss


def evaluate(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    real_dataloader: Iterator[torch.Tensor],
    latent_dataloader: Iterator[torch.Tensor],
    loss_fn: torch.nn.Module,
    normal_label: int,
    anomaly_label: int
) -> tuple:
    generator.eval()
    discriminator.eval()
    
    device = next(generator.parameters()).device
    
    epoch_disc_loss = 0.0
    epoch_gen_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, real in enumerate(real_dataloader):
            real = real.to(device).float()
            batch_size = real.size(0)

            if batch_idx == 0:
                print(f"[DEBUG-EVAL] real shape: {real.shape}, device: {real.device}")
                print(f"[DEBUG-EVAL] Generator device: {next(generator.parameters()).device}")

            real_labels = torch.full((batch_size, 1), normal_label, dtype=torch.float32, device=device)
            fake_labels = torch.full((batch_size, 1), anomaly_label, dtype=torch.float32, device=device)

            z = next(latent_dataloader)
            z = z[:batch_size].to(device).float()

            if batch_idx == 0:
                print(f"[DEBUG-EVAL] z shape: {z.shape}, device: {z.device}")

            fake = generator(z)

            real_output = discriminator(real)
            fake_output = discriminator(fake)
            
            disc_loss_real = loss_fn(real_output, real_labels)
            disc_loss_fake = loss_fn(fake_output, fake_labels)
            disc_loss = disc_loss_real + disc_loss_fake
            
            gen_loss = loss_fn(fake_output, real_labels)
            
            epoch_disc_loss += disc_loss.item()
            epoch_gen_loss += gen_loss.item()
            num_batches += 1
    
    avg_disc_loss = epoch_disc_loss / num_batches
    avg_gen_loss = epoch_gen_loss / num_batches
    
    return avg_gen_loss, avg_disc_loss
