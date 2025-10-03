# Deep Differentially Private Time Series Forecasting

<p align="left">
<img src="https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/_my_direct_uploads/dp_forecasting_thumbnail.png", width="100%">

This is the official implementation of our ICML 2025 Spotlight paper

["Privacy Amplification by Structured Subsampling for Deep Differentially Private Time Series Forecasting"](https://www.cs.cit.tum.de/daml/dp-forecasting/)  
Jan Schuchardt, Mina Dalirrooyfard, Jed Guzelkabaagac, Anderson Schneider, Yuriy Nevmyvaka, Stephan Günnemann.

## Requirements
To install the requirements, execute
```
conda env create -f environment.yaml
```

## Installation
You can install this package via `pip install -e .`

## Usage
In order to reproduce all experiments, you will need need to execute the scripts in `seml/scripts` using the config files provided in `seml/configs`.  
We use the [SLURM Experiment Management Library](https://github.com/TUM-DAML/seml), but you can also directly execute experiments via the `.py` files in `src/experiments`.

After computing all results, you can use the notebooks in `plotting` to recreate the figures from the paper.  

For more details on which config files and plotting notebooks to use for recreating which figure from the paper, please consult [REPRODUCE.md](./REPRODUCE.md).

## Cite
Please cite our paper if you use this code in your own work:

```
@InProceedings{Schuchardt2025_Forecasting,
    author = {Schuchardt, Jan and Dalirrooyfard, Mina and Guzelkabaagac, Jed and Schneider, Anderson and Nevmyvaka, Yuriy and G{\"u}nnemann, Stephan},
    title = {Privacy Amplification by Structured Subsampling for Deep Differentially Private Time Series Forecasting},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2025}
}
```
