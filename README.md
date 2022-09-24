# Code for the Paper "SizeShiftReg: a Regularization Method for Improving Size-Generalization in Graph Neural Networks"
This repository contains code for the paper "SizeShiftReg: a Regularization Method for Improving Size-Generalization in Graph Neural Networks", by Davide Buffelli, Pietro Lio, and Fabio Vandin, accepted at NeurIPS 2022. If you use this repository please cite this paper.
```
@article{buffelli2022sizeshiftreg,
  title={SizeShiftReg: a Regularization Method for Improving Size-Generalization in Graph Neural Networks},
  author={Buffelli, Davide and Lio, Pietro and Vandin, Fabio},
  journal={Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

Part of this library is based on code from: https://github.com/PurdueMINDS/size-invariant-GNNs

### Install
Please install the following dependencies to run this code:
- PyTorch 1.9.0
- `torch-cluster` 1.5.8
- `torch-geometric` 1.7.2
- `torch-scatter` 2.0.5
- `torch-sparse` 0.6.8
- `torch-spline-conv` 1.2.0
- `ray[tune]` 1.1.0

Install the additional dependencies
as follows:

```shell
$ pip install -r requirements.txt
```

A snapshot of the environment used for our experiments can be found in the file `env.txt`.

### Data
Download the data from: `https://zenodo.org/record/6990501#.Yy66sS8w1pQ`
Unzip the folder and put it inside the `data` folder in this repository. The path should be `data/4/..`
The folder `data/` in this library contains the datasets.
The path to this folder needs to be set in the configuration files in the `configs` folder.

### Run Experiments
The `configs` folder contains configuration files for all the models and dataset. To run an experiment simply execute
the command below, sobstituting `{configuration_file_name}` with the name of the configuration file for the experiment
you are interested in. 
__Important__: before running the code below open the configuration file and modify the `data_dir` field so that it points to the path of the `data` folder contained in this repository.

```shell
$ python lightning_modules.py {configuration_file_name}
```

As an example we report the instructions to run experiments for PNA on all datasets, with (the ones with SSR suffix) and without our regularization:
```shell
$ python lightning_modules.py pna_dd
$ python lightning_modules.py pna_dd_ssr
$ python lightning_modules.py pna_proteins
$ python lightning_modules.py pna_proteins_ssr
$ python lightning_modules.py pna_nci1
$ python lightning_modules.py pna_nci1_ssr
$ python lightning_modules.py pna_nci109
$ python lightning_modules.py pna_nci109_ssr
$ python lightning_modules.py pna_deezer
$ python lightning_modules.py pna_deezer_ssr
```
