# Conceptors @ MINDS
*Work in progress.*

This repository contains an implementation of conceptors with echo state networks. It is a work in progress. 

## Roadmap

- [x] Basic implementation of echo state networks
- [x] Basic implementation of "standard" conceptors
   - [x] Store multiple one-dimensional patterns in a network using conceptors, then reproduce their dynamics without input: compute conceptors, then load patterns
   - [x] Basic logic operations to morph different conceptors
- [ ] Add online training of conceptors
- [ ] Basic implementation of autoconceptors
- [ ] Basic implementation of random feature conceptors
- [ ] Basic implementation of diagonal conceptors
- [ ] Building up more complex, modular architectures
   - [ ] Implement the de-noising & classification task
   - [ ] Attending to a signal in a mixture
   - [ ] Equalizing arbitrary filters on signals

## Installation

This repository uses Python 3.8 (you can use [pyenv](https://github.com/pyenv/pyenv) to manage Python versions) and the pip package manager. This works on Linux and MacOS. Once pyenv is installed, you can run 
```
pyenv install 3.8.0
pyenv global 3.8.0
```
to use Python 3.8.

We recommend using [pipenv](https://pipenv.pypa.io/en/latest/) in order to manage your Python libraries and dependencies. 
```bash
pip install pipenv
pipenv update
```

Before installing the packages, you need to download `jaxlib` from source. For Linux, you will need to change the Pipfile to match the followinf filename:
```bash
curl -O https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.1.71-cp38-none-manylinux2010_x86_64.whl
mv jaxlib-0.1.71-cp38-none-manylinux2010_x86_64.whl jaxlib.whl
```
For MacOS:
```bash
curl -O https://storage.googleapis.com/jax-releases/mac/jaxlib-0.1.71-cp38-none-macosx_10_9_x86_64.whl
mv jaxlib-0.1.71-cp38-none-macosx_10_9_x86_64.whl jaxlib.whl
```

Finally, install the packages with:
```
pipenv install
```

## Structure of the Code

The file `esn.py` contains the modules for this project. 

- The `ESN` class contains everything concerning the echo state network.
- The `Optimizer` class is an abstract class for any optimizer that might want to be used for computing the readout weights of the ESN. We provide the two optimizers `LinearRegression` and `RidgeRegression`. Instances of an optimizer class will then be passed to the `ESN.compute_weights` or `ESN.update_weights` methods. 
