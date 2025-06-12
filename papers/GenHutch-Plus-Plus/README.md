# [Optimal Stochastic Trace Estimation in Generative Modeling](https://www.arxiv.org/abs/2502.18808) (AISTATS 2025)

By [Xinyang Liu](https://xinyangatk.github.io)\*<sup>1</sup>, [Hengrong Du](https://hengrongdu.netlify.app)\*<sup>2</sup> , [Wei Deng](https://www.weideng.org)<sup>3</sup>, [Ruqi Zhang](https://ruqizhang.github.io)<sup>1</sup>
\
<sup>1</sup>Purdue University, <sup>2</sup>UC Irvine, <sup>2</sup>Morgan Stanley
\
\*Equal contribution

<a name="why Hutch++?"></a>
## Why Hutch++ in Generative Modeling?
- Lower variance and error in trace estimation compared to Hutchinson trace estimator.
- In generative modeling, Hutch++ is particularly effective for handling ill-conditioned matrices with large condition numbers, which commonly arise when high-dimensional data exhibits a low-dimensional structure. 
- Our practical acceleration technique that balance frequency and accuracy, backed by theoretical guarantees.
- Our analysis demonstrates that Hutch++ leads to generations of higher quality.


<a name="code tructure"></a>
## Code Structure

We provide two separate projects of GruM for three types of graph generation tasks:
- Simulations with Neural ODE model
- Image data modeling with advanced SB-based diffusion models

Each projects consists of the following:
```
ffjord-plus-plus : Code for density estimation of toy distributions with FFJORD++
```

```
sb-fbsde-plus-plus : Code for image data modeling with SB-FBSDE++
```

We provide the details in README.md for each projects.
 
<a name="acknowledgements"></a>
## Acknowledgements
This repository was heavily built off of [FFJORD](https://github.com/rtqichen/ffjord), [SB-FBSDE](https://github.com/ghliu/SB-FBSDE) and [VSDM](https://github.com/WayneDW/Variational_Schrodinger_Diffusion_Model).

<a name="citation"></a>
## Citation
```
@article{liu2025optimal,
  title={Optimal Stochastic Trace Estimation in Generative Modeling},
  author={Liu, Xinyang and Du, Hengrong and Deng, Wei and Zhang, Ruqi},
  journal={arXiv preprint arXiv:2502.18808},
  year={2025}
}
```