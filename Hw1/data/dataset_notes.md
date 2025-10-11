# NASA GES-DISC Knowledge Graph for Link Prediction

## Overview

This dataset contains a knowledge graph of NASA GES-DISC collections, including datasets, data centers, projects, platforms, instruments, science keywords, and publications, along with the relationships between them. It is specifically designed for link prediction tasks.

## Files

- `nodes.csv`: Contains all nodes with their properties.
- `train_edges.csv`: Contains all types of edges used for training.
- `val_links.csv`: Contains `HAS_SCIENCEKEYWORD` edges for validation.
- `test_links.csv`: Contains `HAS_SCIENCEKEYWORD` edges for testing.
- `croissant_metadata.json`: Metadata file describing the dataset.

## Usage

Download the dataset files and use them for machine learning research and link prediction tasks.

## Citation

Please cite this dataset as:

Mehrabian, A. & Gerasimov, I. (2024). NASA GES-DISC Knowledge Graph for Link Prediction. Zenodo. https://doi.org/10.5281/zenodo.11492533

## License

This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
