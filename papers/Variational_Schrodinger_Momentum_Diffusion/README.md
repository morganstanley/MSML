## Variational Schrodinger Critically-damped Langevin Diffusion


### Installation

Following the [link](https://github.com/ghliu/SB-FBSDE), we can install the environment `vscld` using [Anaconda](https://www.anaconda.com/products/individual) as follows
```bash
conda env create --file requirements.yaml python=3
conda activate vscld
```


### Spiral 


#### Vanilla Critically-damped Langevin Diffusion (CLD)

##### beta=5 fails in generation
```python
python main.py --problem-name spiral --diffusion CLD --baseline \
                 --dir spiral_cld --y-scalar 8 --num-itr-dsm 6000 --beta-max 5
```

#### Variational Schrodinger Diffusion Model (VSDM)

##### Using a vanilla fixed beta=5
```python
python main.py --problem-name spiral --diffusion LD --num-stage 60 --num-itr-dsm 100 \
               --dir spiral_vsdm_vanilla --y-scalar 8 --beta-max 5
```

##### Using VPSDE with the same beta schedule as in VSDM (ICML'24)

```python
python main.py --problem-name spiral --diffusion LD --num-stage 60 --num-itr-dsm 100 \ 
               --dir spiral_vsdm_vp --y-scalar 8 --beta-max 10 --beta-r 1.7
```

#### Variational Schrodinger CLD (VSCLD)

```
python main.py --problem-name spiral --diffusion CLD --num-stage 60 --num-itr-dsm 100 \ 
               --dir spiral_vscld --y-scalar 8 --beta-max 5 --damp-ratio 1.0
```

#### Variational Schrodinger ULD (VSULD)

```
python main.py --problem-name spiral --diffusion CLD --num-stage 60 --num-itr-dsm 100 \
               --dir spiral_vsuld_0.7 --y-scalar 8 --beta-max 5 --damp-ratio 0.7
```



### Other Datasets

You can also try Gaussian Mixture (gmm) dataset and Checkerboard dataset using the same setup as the spiral.


### Acknowledgement

https://github.com/ghliu/SB-FBSDE
