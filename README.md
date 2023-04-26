MuseGAN
=========
A Pytorch implementation of MuseGAN

[Check out the generated piano music](#)

:star: Star this project on GitHub — it helps!

[MuseGAN](https://arxiv.org/abs/1709.06298) is a generative model which allows to
generate music.

## Table of content

- [Training](#train)
- [License](#license)
- [Links](#links)

## Training 

See [demo](#) for more details of training process.
* The model components and utils are under `musegan/archs` folder.
* The Midi `torch.utils.dataset` is under `musegan/dataset/data_utils.py`.
* The training Functions and criterions can be found in the `musegan/trainner1 folder


##### `Generator` and `Critic` losses

<a><img src="images/losses.png" align="center" height="500px" width="700px"/></a>

## License

This project is licensed under MIT.

## Links

* [MuseGAN](https://arxiv.org/abs/1709.06298)
