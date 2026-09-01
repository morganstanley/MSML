
# Code for the paper: Prompt-OT: An Optimal Transport Regularization Paradigm for Knowledge Preservation in Vision-Language Model Adaptation
[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)]([https://arxiv.org/abs/2405.03140](https://www.arxiv.org/pdf/2503.08906))


> **Abstract.**
> Vision-language models (VLMs) such as CLIP demonstrate strong performance but struggle when adapted to downstream tasks. Prompt learning has emerged as an efficient and effective strategy to adapt VLMs while preserving their pre-trained knowledge. However, existing methods still lead to overfitting and degrade zero-shot generalization. To address this challenge, we propose an optimal transport (OT)-guided prompt learning framework that mitigates forgetting by preserving the structural consistency of feature distributions between pre-trained and fine-tuned models. Unlike conventional point-wise constraints, OT naturally captures cross-instance relationships and expands the feasible parameter space for prompt tuning, allowing a better trade-off between adaptation and generalization. Our approach enforces joint constraints on both vision and text representations, ensuring a holistic feature alignment. Extensive experiments on benchmark datasets demonstrate that our simple yet effective method can outperform existing prompt learning strategies in base-to-novel generalization, cross-dataset evaluation, and domain generalization without additional augmentation or ensemble techniques. 

# Configuration
Our code is built based on [PromptSRC](https://github.com/muzairkhattak/PromptSRC?tab=readme-ov-file). Thanks for their code!

## Environment Setup

Please refer to [Install.md](https://github.com/muzairkhattak/PromptSRC/blob/main/docs/INSTALL.md)

## Data Prepartion
Please refer to [DATASETS.md]([https://github.com/muzairkhattak/PromptSRC/blob/main/docs/INSTALL.md](https://github.com/muzairkhattak/PromptSRC/blob/main/docs/DATASETS.md))

Just be cautious; in our code, we modify the data dir for imagenet.

## Training and Evaluation


```
#!/bin/bash
DIV=10.0  #lambda in Eq. 16
MYCFG=vit_b16_c2_ep20_batch4_4+4ctx-Copy3
 #training
bash scripts/promptot/base2new_train.sh $dataset 1 $DIV $MYCFG
#evaluation
bash scripts/promptot/reproduce_base2novel_setting.sh $dataset 1 output${DIV}/base2new/train_base/$dataset/shots_16/PromptOT/${MYCFG} $MYCFG $DIV  
    
MYCFG=vit_b16_c2_ep20_batch4_4+4ctx-Copy3
bash scripts/promptot/base2new_train.sh $dataset 2 $DIV $MYCFG
bash scripts/promptot/reproduce_base2novel_setting.sh $dataset 2 output${DIV}/base2new/train_base/$dataset/shots_16/PromptOT/${MYCFG} $MYCFG $DIV
    
MYCFG=vit_b16_c2_ep20_batch4_4+4ctx-Copy3
bash scripts/promptot/base2new_train.sh $dataset 3 $DIV $MYCFG
bash scripts/promptot/reproduce_base2novel_setting.sh $dataset 3 output${DIV}/base2new/train_base/$dataset/shots_16/PromptOT/${MYCFG} $MYCFG $DIV


```


We save the weights for every epoch for further analysis.

## Log process

```
python extract_log_3.py --save_patch output${DIV} --output_path output_csv
```


## Run all dataset

```
sbatch my_script/run_all
```


# Citation
If you find our work is useful in your research, please consider raising a star  :star:  and citing:

```
@article{chen2025prompt,
  title={Prompt-OT: An Optimal Transport Regularization Paradigm for Knowledge Preservation in Vision-Language Model Adaptation},
  author={Chen, Xiwen and Zhu, Wenhui and Qiu, Peijie and Wang, Hao and Li, Huayu and Wu, Haiyu and Sotiras, Aristeidis and Wang, Yalin and Razi, Abolfazl},
  journal={arXiv preprint arXiv:2503.08906},
  year={2025}
}
```


