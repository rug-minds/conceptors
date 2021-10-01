# Conceptors @ MINDS
*Work in progress.*

This repository contains an implementation of conceptors with echo state networks. It is a work in progress. 

## Roadmap

- [ ] Basic implementation of echo state networks
- [x] Basic implementation of "standard" conceptors
   - [ ] Store multiple one-dimensional patterns in a network using conceptors, then reproduce their dynamics without input: compute conceptors, then load patterns
   - [ ] Basic logic operations to morph different conceptors
[ ] Add online training of conceptors
[ ] Basic implementation of autoconceptors
[ ] Basic implementation of random feature conceptors
[ ] Basic implementation of diagonal conceptors
[ ] Building up more complex, modular architectures
   [ ] Implement the de-noising & classification task
   [ ] Attending to a signal in a mixture
   [ ] Equalizing arbitrary filters on signals

## Installation

This repository uses Python 3.8 (you can use [pyenv](https://github.com/pyenv/pyenv) to manage Python versions) and the pip package manager.

We recommend using [pipenv](https://pipenv.pypa.io/en/latest/) in order to manage your Python libraries and dependencies. 

Once you have Python 3.8 installed on your machine, you can install the required packages through pipenv:

```bash
pipenv shell
pipenv install
```

Or without pipenv:

```
pip install -r requirements.txt
```

## Structure of the Code

The file `esn.py` contains the modules for this project. 

- The `ESN` class contains everything concerning the echo state network.
- The `Optimizer` class is an abstract class for any optimizer that might want to be used for computing the readout weights of the ESN. We provide the two optimizers `LinearRegression` and `RidgeRegression`. Instances of an optimizer class will then be passed to the `ESN.compute_weights` or `ESN.update_weights` methods. 
