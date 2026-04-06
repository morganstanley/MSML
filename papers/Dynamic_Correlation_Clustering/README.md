# SPARSE-PIVOT: Dynamic Correlation Clustering for Node Insertions (ICML 2025)

## Experiments

- `experiment_real_graphs.py` — Runs experiments on 4 real-world SNAP graphs (Facebook, Email-Enron, Cit-HepTh, CA-AstroPh).
- `experiment_drift.py` — Runs experiments on the Gas Sensor Array Drift dataset at 5 distance thresholds.
- `mod_pivot.py` — Implementation of the SPARSE-PIVOT algorithm.
- `utils.py` — Shared utilities (OptList data structure, agreement algorithm helpers, clustering evaluation).

## Requirements

- Python 3.10+
- numpy
- networkx
- matplotlib
- seaborn

## Datasets

Download the datasets and place them in a `datasets/` directory before running the experiments.

### SNAP Graphs

From the [Stanford SNAP](https://snap.stanford.edu/data/) collection:

| File | Source |
|------|--------|
| `datasets/musae_facebook.csv` | [Facebook MUSAE](https://snap.stanford.edu/data/facebook-large.html) |
| `datasets/email-Enron.txt.gz` | [Email-Enron](https://snap.stanford.edu/data/email-Enron.html) |
| `datasets/cit-HepTh.txt.gz` | [Cit-HepTh](https://snap.stanford.edu/data/cit-HepTh.html) |
| `datasets/ca-AstroPh.txt.gz` | [CA-AstroPh](https://snap.stanford.edu/data/ca-AstroPh.html) |

### Gas Sensor Array Drift Dataset

From the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations):

- Vergara et al., "Chemical gas sensor drift compensation using classifier ensembles," *Sensors and Actuators B: Chemical*, 2012.
- Rodriguez-Lujan et al., "On the calibration of sensor arrays for pattern recognition using the minimal number of experiments," *Chemometrics and Intelligent Laboratory Systems*, 2014.

Place the 10 batch files as `datasets/drift/batch1.dat` through `datasets/drift/batch10.dat`.

## Usage

```bash
python experiment_real_graphs.py
python experiment_drift.py
```

Each script prints clustering objectives and timing breakdowns, and saves PDF plots to the current directory.
