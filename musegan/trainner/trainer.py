"""Trainer."""

from typing import Iterable

import torch,os
from torch import nn
from tqdm.notebook import tqdm
from .criterion import WassersteinLoss, GradientPenalty


class Trainer():
    """Trainer."""

    def __init__(
        self,
        generator,#generator
        critic,#discriminator
        g_optimizer,#generator nn.optim
        c_optimizer,#discriminator nn.optim
        ckpt_path, #checkpoint path
        device: str = "cuda:0", #torch device
        
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
    def save_ckp(
        self,
        state,
        checkpoint_path,
        ) -> None:
            """
            state: type dict
            checkpoint_path: path to save checkpoint
            """
            # save checkpoint data to the path given, checkpoint_path
            torch.save(state, checkpoint_path)
          
    """ load model """
    def load_ckp(
        self,
        checkpoint_fpath,
        model,
        optimizer,
        ) -> None:
            """
            checkpoint_path: path to save checkpoint
            model: model that we want to load checkpoint parameters into       
            optimizer: optimizer we defined in previous training
            """
            # load check point
            checkpoint = torch.load(checkpoint_fpath)
            # initialize state_dict from checkpoint to model
            model.load_state_dict(checkpoint['state_dict'])
            # initialize optimizer from checkpoint to optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
            # initialize valid_loss_min from checkpoint to valid_loss_min
            # return model, optimizer, epoch value, min validation loss 
            return model, optimizer, checkpoint['epoch']
    
    """Training Loop Function"""
    def train(
        self,
        dataloader: Iterable,
        epochs: int = 500,
        batch_size: int = 64,
        repeat: int = 5,
        #(deprecated)display_step: int = 10,
        melody_groove: int = 5,
        save_checkpoint: bool = True,
    ) -> None:
        os.makedirs(self.ckpt_path, exist_ok=True)
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
                    e_cfloss += b_cfloss / len(train_loader)
                    e_crloss += b_crloss / len(train_loader)
                    e_cploss += b_cploss / len(train_loader)
                    e_closs += b_closs / len(train_loader)
                    #SAVE DISC MODEL STATE DICT
                    if save_checkpoint:
                        checkpoint = {
                          'epoch': epoch+1,
                          'state_dict': self.critic.state_dict(),
                          'optimizer': self.c_optimizer.state_dict(),
                          }
                        self.save_ckp(checkpoint, os.path.join(self.ckpt_path, 'museGAN_netD-{}.pth'.format(epoch)))
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
                    e_gloss += b_gloss.item() / len(train_loader)
            """
            Append Losses
            """
            self.data['gloss'].append(e_gloss)
            self.data['closs'].append(e_closs)
            self.data['cfloss'].append(e_cfloss)
            self.data['crloss'].append(e_crloss)
            self.data['cploss'].append(e_cploss)
            #SAVE GEN MODEL STATE DICT
            if save_checkpoint:
                checkpoint = {
                  'epoch': epoch+1,
                  'state_dict': self.generator.state_dict(),
                  'optimizer': self.g_optimizer.state_dict(),
                  }
                self.save_ckp(checkpoint, os.path.join(self.ckpt_path, 'museGAN_netG-{}.pth'.format(epoch)))
            """
                Loss Statistics
            """
            train_loader.set_postfix(losses='Epoch: {epoch} \tGenerator loss: {e_gloss:.3f}\tCritic loss: {e_closs:.3f}')
            train_loader.set_postfix(penalties='tfake: {e_cfloss:.3f}\treal: {e_crloss:.3f}\tpenalty: {e_cploss:.3f}')
            torch.cuda.empty_cache()
            #(deprecated)if epoch % display_step == 0:
             #(deprecated)   print(f"Epoch {epoch}/{epochs} | Generator loss: {e_gloss:.3f} | Critic loss: {e_closs:.3f}")
              #(deprecated)  print(f"(fake: {e_cfloss:.3f}, real: {e_crloss:.3f}, penalty: {e_cploss:.3f})")
