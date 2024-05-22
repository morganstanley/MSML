# Dynamic System Identification
```
@article{zheng2024constrained,
  title={Constrained Exploration via Reflected Replica Exchange Stochastic Gradient Langevin Dynamics},
  author={Zheng, Haoyang and Du, Hengrong and Feng, Qi and Deng, Wei and Lin, Guang},
  journal={arXiv preprint arXiv:2405.07839},
  year={2024}
}
```

![image](https://github.com/haoyangzheng1996/r2SGLD/assets/38525155/786c2e29-ff1f-4625-b8b6-dc9d0ea8e169)


# Prerequisites
Please refer to "env_dynamic_multimodal.yml" 

# Usage
For SGLD, please run:
```
python3 lorenz_sgld.py
```
![reflect_sgld](https://github.com/haoyangzheng1996/r2SGLD/assets/38525155/22085320-2287-4763-8fd5-1f8d1349d5d0)


For cycSGLD, please run:
```
python3 lorenz_cycsgld.py
```
![reflect_cycsgld](https://github.com/haoyangzheng1996/r2SGLD/assets/38525155/728340a3-d130-4184-9639-5e9ac6880f0f)


For SGLD, please run:
```
python3 lorenz_resgld.py
```
![reflect_resgld](https://github.com/haoyangzheng1996/r2SGLD/assets/38525155/b97d315b-5f6a-40d2-aa06-76f95be41767)


You can also run:
```
source main_lorenz.sh
```


## References:

Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz. "[Discovering governing equations from data by sparse identification of nonlinear dynamical systems.](https://www.pnas.org/doi/full/10.1073/pnas.1517384113)" PNAS 2016.
   
