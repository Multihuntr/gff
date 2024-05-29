# Global Flood Forecasting (GFF)

![GFF global map](./gff_map.png)

A globally distributed dataset for learning flood forecasting featuring both generated and labelled flood maps.

Download here: TODO

# This repository

Provides code to:

1. **Use GFF**: Train and evaluate models on GFF dataset.
2. **Generate data**: Generate GFF-like data from original sources.

## Use GFF

To simply train and evaluate models on GFF, just build the conda environment. E.g.

```python
conda env create -p envs/flood --file environment.yml
```

And run `train.py` or `evaluate.py`. Look in `configs/*` for how to provide configuration options. E.g.:

```python
python train.py configs/two_recunet.yml -o data_folder=path/to/gff
```

Will create a `./runs` folder, and save a model into it.

## Generate new data

To generate new data using GFF's methodology/code, see `./HOWTO-GENERATE.md`

# Cite

To cite us, use:

```
TBD
```
