# Code for: <i>Learning to Strategically Acquire Resources in Competition</i>
This paper provides a generalized game theory model for competitive resource acquisition, proves well-behaved equilibrium structure even under imperfect information, and studies when simultaneous learning dynamics converge to that equilibrium even without a common-knowledge prior. 

Authors: [Safwan Hossain](https://safwanhossain.github.io/), [Mirah Shi](https://www.seas.upenn.edu/~mirahshi/), [Andrew Bennett](https://awbennett.net/), [Neil Chriss](https://en.wikipedia.org/wiki/Neil_Chriss), [Michael Kearns](https://www.cis.upenn.edu/~mkearns/), [Anderson Schneider](https://scholar.google.com/citations?user=KLyaFtUAAAAJ&hl=en), [Yuriy Nevmyvaka](https://scholar.google.com/citations?user=Hui4EIcAAAAJ&hl=en)

Link to Paper: https://arxiv.org/pdf/2606.06882 

## main.py
Contains algorithms for computing best response and equilibrium.

## cost_models.py
Contains the code used to compute prices within our model. Please don't compute prices elsewhere.Always use get_price_vector whenever prices need to be computed.

## misc.py
Old code that may be useful at some point later. Safely ignore

## plots.py
For generating plots.

## spne.py
Algorithm to compute SPNE in the Markovian game
