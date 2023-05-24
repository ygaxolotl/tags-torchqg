Torch QG
=== 

This code has been modified from the base code at: https://github.com/hrkz/torchqg/
``` 
Note: The base code at (https://github.com/hrkz/torchqg/) is not anonymized. But anonymization of the NeurIPS23 paper still holds, because the linked base repository is a public repository that is not part of the contribution. The authors of the linked code might or might not be the same authors of the NeurIPS23 paper. 
```
## Usage

See `main.py` in the root folder for a simulation example based on [Graham et al. 2013](https://doi.org/10.1016/j.ocemod.2013.01.004). A notebook with a simple end-to-end trained parametrization might appear later.

## Research

The code was initially developped for subgrid-scale (SGS) parametrization learning, in particular with an end-to-end approach, i.e. where gradient of the forward solver is available. The first reference describing the setup can be accessed [here](https://arxiv.org/pdf/2111.06841.pdf).

## Install
```
git clone git@github.com:ygaxolotl/tags-torchqg.git
cd tags-torchqg
git checkout paper
```