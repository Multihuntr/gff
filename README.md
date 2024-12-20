# Global Flood Forecasting (GFF)

![GFF global map](./gff_map.png)

A globally distributed dataset for learning flood forecasting featuring both generated and labelled flood maps.

Dataset download here: [Zenodo](https://zenodo.org/records/14184289)

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
@article{victor2024off,
  title={Off to new Shores: A Dataset \& Benchmark for (near-) coastal Flood Inundation Forecasting},
  author={Victor, Brandon and Letard, Mathilde and Naylor, Peter and Douch, Karim and Long{\'e}p{\'e}, Nicolas and He, Zhen and Ebel, Patrick},
  journal={arXiv preprint arXiv:2409.18591},
  year={2024}
}
```

# Acknowledgements

This project was initiated at [Φ-lab](https://philab.esa.int/) at ESRIN, when @Multihuntr (Brandon Victor) temporarily joined as a visiting researcher. Thanks to [SmartSatCRC](https://smartsatcrc.com/) for funding the visit, and to Φ-lab for hosting the visit.
