Code repository: https://github.com/alluly/diffusion-excursion-point-process

Reference: [Inference and Sampling of Point Processes from Diffusion
Excursions](https://proceedings.mlr.press/v216/hasan23a.html)


Ali Hasan, Yu Chen, Yuting Ng, Mohamed Abdelghani, Anderson Schneider,
and Vahid Tarokh in UAI 2023 (**Spotlight**)


Abstract: Point processes often have a natural interpretation with
respect to a continuous process. We propose a point process
construction that describes arrival time observations in terms of the
state of a latent diffusion process. In this framework, we relate the
return times of a diffusion in a continuous path space to new arrivals
of the point process. This leads to a continuous sample path that is
used to describe the underlying mechanism generating the arrival
distribution. These models arise in many disciplines, such as
financial settings where actions in a market are determined by a
hidden continuous price or in neuroscience where a latent stimulus
generates spike trains. Based on the developments in Itô’s excursion
theory, we propose methods for inferring and sampling from the point
process derived from the latent diffusion process. We illustrate the
approach with numerical examples using both simulated and real
data. The proposed methods and framework provide a basis for
interpreting point processes through the lens of diffusions.
