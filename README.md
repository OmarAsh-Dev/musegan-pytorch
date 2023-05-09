MuseGAN
=========
A Pytorch implementation of MuseGAN
Only linux BSD support due to SharedArray usage


![PyPI - License](https://img.shields.io/pypi/l/musegan)
[![PyPI](https://img.shields.io/pypi/v/musegan)](https://pypi.org/project/musegan/)
![PyPI - Downloads](https://img.shields.io/pypi/dw/musegan)

[MuseGAN](https://arxiv.org/abs/1709.06298) is a generative model which allows to
generate music.

## Table of content

- [Training](#train)
- [License](#license)
- [Links](#links)

## Training 

See this [colab](https://colab.research.google.com/drive/1NF2t1dvqxeblZfd7BL4Gfn4SW-xEzgGg?authuser=3#scrollTo=9bj_FWvAArPI)  notebook for more details of training process.
* The model components and utils are under `musegan/archs` folder.
* The Midi `torch.utils.dataset` is under `musegan/dataset/data_utils.py`.
* The training Functions and criterions can be found in the `musegan/trainner` folder




## License

This project is licensed under MIT.

## Links

* [MuseGAN](https://arxiv.org/abs/1709.06298)
