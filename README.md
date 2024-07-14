# PyJama: Differentiable Jamming and Anti-Jamming with NVIDIA Sionna

PyJama is an open-source Python library for simulation and training of jamming and anti-jamming algorithms.
It is built on top of [Sionna](https://nvlabs.github.io/sionna) and [TensorFlow](https://www.tensorflow.org).

The official website of this library can be found [here](http://pyjama.ethz.ch), documentation and example code is located [here](https://huoxiaoyao.github.io/pyjama).


## Installation

For GPU support, please install and verify the required GPU drivers (see the [official tensorflow guide](https://www.tensorflow.org/install/pip) for details).

For now, PyJama can only be installed from source. In the future, this package will also be released on PyPi, so that you may install it directly via pip.

### Installation from Source

First, please clone this repository to your local machine.
You may then install the requirements for PyJama by executing (preferably in a virtual environment):
```
pip install -r requirements.txt
```
inside the repository root folder.

If you want to build the documentation, you can install the additional requirements using
```
pip install -r requirements_doc.txt
```
but this is not necessary for only using PyJama.

Lastly, install PyJama by running
```
make install
```
in the root folder.

Afterwards, test the installation by running
```
python -c 'import pyjama; print pyjama.__version__'
```

## Licence and Citation
PyJama is licensed under APACHE-2.0 license, as found in the [LICENSE]() file.

When using PyJama, you _must_ cite our paper:
```
@inpreparation{ulbricht2024pyjama,
  title={{PyJama}: Differentiable Jamming and Anti-Jamming with {NVIDIA Sionna}},
  author={Ulbricht, Fabian and Marti, Gian and Wiesmayr, Reinhard and Studer, Christoph},
  year={2024}
}
```

## Other information
All simulations in this repository are implemented with [NVIDIA Sionna](https://nvlabs.github.io/sionna/) Release v0.15.1.
Older versions might be compatible, but are not actively supported.
