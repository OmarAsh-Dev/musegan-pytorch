MuseGAN
=========
![PyPI - License](https://img.shields.io/pypi/l/musegan)
[![PyPI](https://img.shields.io/pypi/v/musegan)](https://pypi.org/project/musegan/)
![PyPI - Downloads](https://img.shields.io/pypi/dw/musegan)

- A Pytorch implementation of MuseGAN
- ***Note :*** Only linux BSD support due to SharedArray usage which is a linux only pip package




[MuseGAN](https://arxiv.org/abs/1709.06298) is a generative model which allows to
generate music.

Usage
=======
The musegan package contains all the schema needed to train and generate your own music in MIDI format.
There are two ways of using the musegan package;

1. Using the PYPI package.
   - To use it use the command: `pip install musegan` to install all the necessary packages

2. Building from the source 
   - First clone the repository to your local directory with `git`
   - Open your local Terminal/Command Shell and run the following commands
    ```shell
    cd musegan-pytorch # change directory to the cloned repository
    #use any of the following some will work dependent on your operating system
    #try
    python3 setup.py develop
    #or
    pip install .
    #or
    pip install -e
    ```
    - To test run the following in your python terminal;
    ```py
    >> import musegan
    >> print(musegan__version__)
    # which should return the latest musegan version according to the time you read this
    0.0.9
    ```
    Now you should be able to use zthe musegan packages
    
## Table of content

- [Training](https://github.com/cliffordkleinsr/musegan-pytorch/edit/dev/README.md#training)
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
