"""Trainer."""

from typing import Iterable

import torch
from torch import nn
from tqdm.notebook import tqdm
from .criterion import WassersteinLoss, GradientPenalty


class Trainer():
    """Trainer."""

    def __init__(
        self,
        generator,
        critic,
        g_optimizer,
        c_optimizer,
        device: str = "cuda:0",
    ) -> None:
        """Initialize."""
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)
        self.device = device

    def train(
        self,
        dataloader: Iterable,
        epochs: int = 500,
        batch_size: int = 64,
        repeat: int = 5,
        #display_step: int = 10,
        melody_groove: int = 5,
    ) -> None:
        """
        Why rand/randn?
                - First, as you see from the documentation numpy.random.randn
                generates samples from the normal distribution,
                while numpy.random.rand from a uniform distribution (in the range [0,1)).
        """
        """Start training process."""
        self.alpha = torch.rand((batch_size, 1, 1, 1, 1)).requires_grad_().to(self.device)
        self.data = {
            "gloss": [],
            "closs": [],
            "cfloss": [],
            "crloss": [],
            "cploss": [],
        }
        for epoch in range(epochs):
            e_gloss = 0
            e_cfloss = 0
            e_crloss = 0
            e_cploss = 0
            e_closs = 0
            with tqdm(dataloader, unit= 'it') as train_loader:
                for real in train_loader:
                    real = real.to(self.device)
                    # Train Critic
                    b_closs = 0
                    b_cfloss = 0
                    b_crloss = 0
                    b_cploss = 0
                    for _ in range(repeat):
                        # Very important note
                        # chords shape: (batch_size, z_dimension)
                        # style shape: (batch_size, z_dimension)
                        # melody shape: (batch_size, n_tracks, z_dimension)
                        # groove shape: (batch_size, n_tracks, z_dimension)
                        """
                        # create random `noises`
                        """
                        cords = torch.randn(batch_size, 32).to(self.device)
                        style = torch.randn(batch_size, 32).to(self.device)
                        melody = torch.randn(batch_size, melody_groove, 32).to(self.device)
                        groove = torch.randn(batch_size, melody_groove, 32).to(self.device)
                        """
                        # forward to generator
                        """
                        self.c_optimizer.zero_grad()
                        with torch.no_grad():
                            fake = self.generator(cords, style, melody, groove).detach()
                        """
                        # mix `real` and `fake` melody
                        """
                        realfake = self.alpha * real + (1. - self.alpha) * fake
                        """
                        # get critic's `fake` loss, # get critic's `real` loss,
                        # get critic's penalty
                        """
                        fake_pred = self.critic(fake)
                        real_pred = self.critic(real)
                        realfake_pred = self.critic(realfake)
                        fake_loss = self.c_criterion(fake_pred, - torch.ones_like(fake_pred))#critic's `fake` loss
                        real_loss = self.c_criterion(real_pred, torch.ones_like(real_pred))#critic's `real` loss
                        penalty = self.c_penalty(realfake, realfake_pred)#critic's penalty
                        """
                        # sum up losses
                        """
                        closs = fake_loss + real_loss + 10 * penalty
                        """
                        # retain graph
                        """
                        closs.backward(retain_graph=True)
                        """
                        # update critic parameters
                        """
                        self.c_optimizer.step()
                        """
                        # devide by number of critic updates in the loop (5)
                        """
                        b_cfloss += fake_loss.item() / repeat
                        b_crloss += real_loss.item() / repeat
                        b_cploss += 10 * penalty.item() / repeat
                        b_closs += closs.item() / repeat
                    """
                    Append the critic losses
                    """
                    e_cfloss += b_cfloss / len(dataloader)
                    e_crloss += b_crloss / len(dataloader)
                    e_cploss += b_cploss / len(dataloader)
                    e_closs += b_closs / len(dataloader)
                    
                    # Train Generator
                    self.g_optimizer.zero_grad()
                    # Very important note
                    # chords shape: (batch_size, z_dimension)
                    # style shape: (batch_size, z_dimension)
                    # melody shape: (batch_size, n_tracks, z_dimension)
                    # groove shape: (batch_size, n_tracks, z_dimension)
                    """
                    # create random `noises`
                    """
                    cords = torch.randn(batch_size, 32).to(self.device)
                    style = torch.randn(batch_size, 32).to(self.device)
                    melody = torch.randn(batch_size, melody_groove, 32).to(self.device)
                    groove = torch.randn(batch_size, melody_groove, 32).to(self.device)
                    """
                    # forward to generator
                    """
                    fake = self.generator(cords, style, melody, groove)
                    """
                    # forward to critic (to make prediction)
                    """
                    fake_pred = self.critic(fake)
                    """
                    # get generator loss (idea is to fool critic)
                    """
                    b_gloss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                    b_gloss.backward()
                    """
                    # update critic parameters
                    """
                    self.g_optimizer.step()
                    e_gloss += b_gloss.item() / len(dataloader)
            """
            Append Losses
            """
            self.data['gloss'].append(e_gloss)
            self.data['closs'].append(e_closs)
            self.data['cfloss'].append(e_cfloss)
            self.data['crloss'].append(e_crloss)
            self.data['cploss'].append(e_cploss)
            """
                Loss Statistics
            """
            train_loader.set_postfix(losses ='Epoch: {epoch} \tGenerator loss: {:.3f} \tCritic loss: {:.3f} \tfake: {:.3f} \treal: {:.3f} \tpenalty: {:.3f}'.format(e_gloss, e_closs, e_cfloss, e_crloss, e_cploss))
            #if epoch % display_step == 0:
             #   print(f"Epoch {epoch}/{epochs} | Generator loss: {e_gloss:.3f} | Critic loss: {e_closs:.3f}")
              #  print(f"(fake: {e_cfloss:.3f}, real: {e_crloss:.3f}, penalty: {e_cploss:.3f})")
