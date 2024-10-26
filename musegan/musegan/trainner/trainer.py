"""Trainer."""

from typing import Iterable
import torch, os
from torch import nn
from tqdm.notebook import tqdm
from .criterion import WassersteinLoss, GradientPenalty


class Trainer():
    """Trainer."""

    def __init__(
        self,
        generator,  # generator
        critic,  # discriminator
        g_optimizer,  # generator nn.optim
        c_optimizer,  # discriminator nn.optim
        ckpt_path,  # checkpoint path
        device: str = "cuda:0",  # torch device
    ) -> None:
        """Initialize."""
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)
        self.ckpt_path = ckpt_path
        self.device = device

    """Save model"""
    def save_ckp(self, state, checkpoint_path) -> None:
        """Save checkpoint."""
        torch.save(state, checkpoint_path)

    """Load model"""
    def load_ckp(self, checkpoint_fpath, model, optimizer) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']

    """Training Loop Function"""
    def train(
        self,
        dataloader: Iterable,
        start_epoch: int = 0,
        epochs: int = 500,
        batch_size: int = 64,
        repeat: int = 5,
        melody_groove: int = 4,
        save_checkpoint: bool = True,
        model_name: str = "museGAN"
    ) -> None:
        os.makedirs(self.ckpt_path, exist_ok=True)

        """Start training process."""
        self.alpha = torch.rand((batch_size, 1, 1, 1, 1)).requires_grad_().to(self.device)
        self.data = {
            "gloss": [],
            "closs": [],
            "cfloss": [],
            "crloss": [],
            "cploss": [],
        }

        for epoch in range(start_epoch, epochs):
            e_gloss = 0
            e_cfloss = 0
            e_crloss = 0
            e_cploss = 0
            e_closs = 0

            with tqdm(dataloader, unit='it') as train_loader:
                for real in train_loader:
                    real = real.to(self.device)
                    # Train Critic
                    b_closs = 0
                    b_cfloss = 0
                    b_crloss = 0
                    b_cploss = 0
                    for _ in range(repeat):
                        cords = torch.randn(batch_size, 32).to(self.device)
                        style = torch.randn(batch_size, 32).to(self.device)
                        melody = torch.randn(batch_size, melody_groove, 32).to(self.device)
                        groove = torch.randn(batch_size, melody_groove, 32).to(self.device)
                        
                        self.c_optimizer.zero_grad()
                        with torch.no_grad():
                            fake = self.generator(cords, style, melody, groove).detach()
                        
                        realfake = self.alpha * real + (1. - self.alpha) * fake

                        fake_pred = self.critic(fake)
                        real_pred = self.critic(real)
                        realfake_pred = self.critic(realfake)
                        fake_loss = self.c_criterion(fake_pred, -torch.ones_like(fake_pred))
                        real_loss = self.c_criterion(real_pred, torch.ones_like(real_pred))
                        penalty = self.c_penalty(realfake, realfake_pred)

                        closs = fake_loss + real_loss + 10 * penalty
                        closs.backward(retain_graph=True)
                        self.c_optimizer.step()

                        b_cfloss += fake_loss.item() / repeat
                        b_crloss += real_loss.item() / repeat
                        b_cploss += 10 * penalty.item() / repeat
                        b_closs += closs.item() / repeat

                    e_cfloss += b_cfloss / len(train_loader)
                    e_crloss += b_crloss / len(train_loader)
                    e_cploss += b_cploss / len(train_loader)
                    e_closs += b_closs / len(train_loader)
                    
                    # Train Generator
                    self.g_optimizer.zero_grad()
                    cords = torch.randn(batch_size, 32).to(self.device)
                    style = torch.randn(batch_size, 32).to(self.device)
                    melody = torch.randn(batch_size, melody_groove, 32).to(self.device)
                    groove = torch.randn(batch_size, melody_groove, 32).to(self.device)

                    fake = self.generator(cords, style, melody, groove)
                    fake_pred = self.critic(fake)
                    b_gloss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                    b_gloss.backward()
                    self.g_optimizer.step()
                    e_gloss += b_gloss.item() / len(train_loader)

                train_loader.set_postfix(losses=f'Epoch: {epoch} | Generator loss: {e_gloss:.3f} | Critic loss: {e_closs:.3f} | fake: {e_cfloss:.3f} | real: {e_crloss:.3f} | penalty: {e_cploss:.3f}')
            
            self.data['gloss'].append(e_gloss)
            self.data['closs'].append(e_closs)
            self.data['cfloss'].append(e_cfloss)
            self.data['crloss'].append(e_crloss)
            self.data['cploss'].append(e_cploss)
            
            # Save checkpoint every 500 epochs
            if save_checkpoint and (epoch + 1) % 500 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.generator.state_dict(),
                    'optimizer': self.g_optimizer.state_dict(),
                }
                self.save_ckp(checkpoint, os.path.join(self.ckpt_path, f'{model_name}_Net_G-{epoch}.pth'))

                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.critic.state_dict(),
                    'optimizer': self.c_optimizer.state_dict(),
                }
                self.save_ckp(checkpoint, os.path.join(self.ckpt_path, f'{model_name}_Net_D-{epoch}.pth'))

            torch.cuda.empty_cache()

