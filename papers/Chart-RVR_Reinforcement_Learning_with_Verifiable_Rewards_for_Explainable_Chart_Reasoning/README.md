# ChartRL: Improving Chart understanding in VLMs 

Repo for RL post-training on Chart data. We present a small 3 billion chart understanding model which outputs answers AND rationales. Currently the model is based on Qwen2.5VL (3 billion).

The design hinges on training VLMs using GRPO with verifiable rewards.
The training data is sampled from train sets of - ChartQA, PlotQA, and ChartFC (in-domain).


Huge thanks to the team at Morgan Stanley for their support!


Find our Chart-RVR-3B models and associated datasets on [Huggingface](https://huggingface.co/collections/sanchit97/chart-rvr-68aaac32a2745bc653f581a1).



## Main Benchmark Results

| Approach                           | Exp? | ChartQA | PlotQA | ChartFC | EvoChart | ChartQAPro | 
|------------------------------------|------|---------|--------|---------|----------|------------|
| **Direct Prompting**               |      |         |        |         |          |            |
| Q2.5VL-Ins                         | ✗    | 82.0    | 80.5   | 74.4    | 48.72    | 25.7       |
| **Explainable Models - with Rationales** |      |         |        |         |          |      |            
| Q2.5VL-Ins (CoT)                   | ✔    | 73.12   | 52.72  | 69.20   | 29.6     | 15.80      | 
| ChartGemma                         | ✔    | 76.44   | 33.28  | 70.33   | 36.96    | 10.93      | 
| **Fine-tuned Models - with Rationales** |      |         |        |         |          |       | 
| Q2.5VL-SFT                         | ✔    | 83.08   | 74.18  | 77.30   | 46.08    | 23.56      | 
| Q2.5VL-DPO                         | ✔    | 75.42   | 53.86  | 70.1    | 34.8     | 15.95      | 
| Q2.5VL-Ins (F+L+Tasks)             | ✔    | 81.8    | 76.24  | 63.85   | 51.68    | 27.66      | 
| **Chart-RVR-3B (Ours)**            | ✔    | 84.56   | **78.68** | 77.62   | 53.36    | 28.38   |  
| **Data Curated Fine-tuned Models - with Rationales** |      |         |        |         |          |            |            |
| Q2.5VL-SFT-Hard                    | ✔    | 84.28   | 75.54  | 77.90   | 49.36    | 23.20      | 
| **Chart-RVR-3B-Hard (Ours)**       | ✔    | **85.76** | 77.9   | **80.07** | **54.24** | **28.64** | 


## Running Evals

Run inference on ChartQA with Chain-of-Thought:
```python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartqa-src --cot True```

Run inference on ChartQA with GRPO trained models:
```python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartqa-src --cot True --grpo-lora True```

Run inference on ChartQA with SFT trained models:
```python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartqa-src --cot True --sft-lora True```

Ensure path to models correspond to checkpoints from [Chart-RVR](https://huggingface.co/sanchit97/chart-rvr-3b) or [Chart-RVR-Hard](https://huggingface.co/sanchit97/chart-rvr-hard-3b)


To ease computation, we have provided scripts which can directly be executed on SLURM in ```./eval-scripts```
For instance to run GRPO on all datasets, use: ```sbatch ./eval-scripts/grpo_all_eval.sh```. This runs evaluation on all datasets and stores it in ```./logs```.

## Training Models

### SFT
We utilize the great work [here](https://github.com/2U1/Qwen2-VL-Finetune) to fine-tune models.


### Chart-RVR (GRPO-based) 
To directly run training, we recommend first double checking TRL version. Next double check the ```num_processes``` in ```./deepspeed_zero3.yaml``` is set to 4 (recommended). For any other combination, there might be OOM errors with default completions, prompt size, etc. 

Then, simply run on the SLURM cluster ```sbatch grpo.sh```. For running directly on local GPUs, run:
```accelerate launch --config_file=deepspeed_zero3.yaml main.py --mode grpo --vlm-name qwen2-5-3b --dataset-name chartqa-src```

Note: The datset name is only placeholder for the eval-script. It is mandatory to run the dataset generation script first.