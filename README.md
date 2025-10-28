# PyFowl
**The Python code for Fast Observables in Weak Lensing**  

- EFT one-loop predictions for 2D angular projected two-point functions: 
	- cosmic shear
	- galaxy-galaxy lensing
	- galaxy clustering
- Dark Energy Survey Y3 3x2pt likelihood with EFT predictions

[![](https://img.shields.io/badge/arXiv-2510.xxxxx%20-red.svg)](https://arxiv.org/abs/2510.xxxxx)
[![](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/pierrexyz/pyfowl/blob/master/LICENSE)


## Installation

```bash
git clone https://github.com/pierrexyz/pyfowl.git
cd pyfowl
pip install -e .
```


## Running with MontePython
To run with [MontePython 3](https://github.com/brinckmann/montepython_public), once PyFowl is installed as above,  

* Copy the likelihood folder [montepython/likelihoods/eftdes](montepython/likelihoods/eftdes) to your working MontePython repository: `montepython_public/montepython/likelihoods/`  
* Copy the data folder [data/eftdes](data/eftdes) to your working MontePython data folder: `montepython_public/data/`  
* Run the DES-Y3 3x2pt EFT likelihood with the input param file [montepython/input/maglim_noz56.param](montepython/input/maglim_noz56.param)  

* Posterior covariances for Metropolis-Hasting Gaussian proposal (in MontePython format) can be found [here](montepython/chains).  


## Attribution
* Devs:
    * [Pierre Zhang](mailto:pierrexyz@protonmail.com)
    * [Guido D'Amico](mailto:damico.guido@gmail.com) 
* License: MIT

When using PyFowl in a publication, please acknowledge the code by citing the following paper:  
> G. D’Amico, A. Refregier, L. Senatore, and P. Zhang, "The cosmological analysis of DES 3$\times$2pt data from the Effective Field Theory of Large-Scale Structure", [2510.xxxxx](https://arxiv.org/abs/2510.xxxxx)

The BibTeX entry is:
```
...
```

When using the Dark Energy Survey Year 3 data products in a publication, please refer to [https://des.ncsa.illinois.edu/releases/y3a2](https://des.ncsa.illinois.edu/releases/y3a2) for acknowledgements. 

