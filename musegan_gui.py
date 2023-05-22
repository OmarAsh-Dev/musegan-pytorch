from musegan.archs import TemporalNetwork, BarGenerator, MuseCritic, MuseGenerator, initialize_weights
from musegan.dataset import LPDDataset, postprocess
from musegan.trainner import Trainer

import os, numpy as np, torch, glob as glob, argparse
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from gooey import Gooey

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_networks(
    n_bars,
    n_tracks,
) -> None:
    #temporal network
    tempnet = TemporalNetwork(n_bars=n_bars)
    #bargenerator
    bargenerator = BarGenerator(
        z_dimension=32,
        hid_features=1152,
        hid_channels=192,
        n_steps_per_bar=48,
        n_pitches=84,
    )
    #GAN Generator
    generator = MuseGenerator(
        z_dimension=32,
        hid_channels=192 * 2,
        hid_features=1152,
        out_channels=1,
        n_tracks=n_tracks,
        n_bars=n_bars,
        n_steps_per_bar=48,
        n_pitches=84,
    )
    #GAN Discriminator
    critic = MuseCritic(
        hid_channels=128,
        n_tracks=n_tracks,
        n_bars=n_bars,
        n_steps_per_bar=48,
        n_pitches=84,
    )
    return tempnet, bargenerator, generator, critic

def dataLoader(
    path, #->posix path
    batch_soize,
    num_wakas,
) -> None:
    dataset = LPDDataset(path)
    dataloader = Dataloader(
        dataset, batch_size=batch_soize,
        shuffle=True, drop_last=True, num_workers=num_wakas
        )
    return dataloader

@Gooey(program_name='Musegan', image_dir='images') 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-nt',
        '--n_tracks',
        type=int,
        default=5,
        help='The number of tracks in your data')

    parser.add_argument(
        '-nb',
        '--n_bars',
        type=int,
        default=8,
        help='The number of bars you want from your data')
    parser.add_argument(
        '-ne',
        '--epochs',
        type=int,
        default=10,
        help='The number of iterations')
    parser.add_argument('--dataset', type=argparse.FileType('r'), help='Dataset folder ')
    parser.add_argument('--ckpt', type=str, default='ckpt_path', help='Checkpoint path')              
    parser.add_argument('-b', '--batch_size', type=int,
                         default=4, help='Batch Size of your Data loader')
    parser.add_argument('-w', '--num_workers', type=int,
                         default=4, help='The number of workers for your specified Data loader')
     
     
    args = parser.parse_args()
    #bild net
    tempnet, bargenerator, generator, critic = build_networks(args.n_bars, args.n_tracks)
    #bild DL
    glob = glob.glob(os.path.join(args.dataset, *npy))
    file = glob[0]
                
    dataloader = dataLoader(file, args.batch_size, args.num_workers)

    #generator model and optimizer
    generator = generator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.9))

    # discriminator model and optimizer
    critic = critic.to(device)
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001, betas=(0.5, 0.9))

    #initialize weights
    generator = generator.apply(initialize_weights)
    critic = critic.apply(initialize_weights)

    cpkt_path = args.ckpt
    #os.makedirs(args.ckpt, exist_ok=True)
    trainer = Trainer(generator, critic, g_optimizer, c_optimizer, cpkt_path)
    
    #Train
    trainer.train(dataloader, epochs=args.epochs, batch_size=args.batch_size, melody_groove= args.n_tracks)

    

if __name__ == '__main__':
    main()
