## LLM-speedrun-specific Conductor guidance

The metric is validation bits-per-byte (val BPB) under a wall-clock + parameter budget. Specific failure modes:

- **Budget violations slipping through.** Some experiments exceed the param budget after `torch.compile` injects buffers, or exceed the wall-clock budget by 1-2% due to overhead. Confirm the experiment's eval framework rejected over-budget runs before annotating. If not, mark `quarantined`.
- **Val-set contamination.** Pre-tokenized caches or shared dataloaders can mix val tokens into train. A val-BPB that is suspiciously below the literature floor on a similar model size is usually contamination. Direct the strategist to dump a few val sequences and grep the train shards.
- **Single-seed champions.** LLM pretraining is high-variance. Treat a sub-1% improvement on one seed as noise unless reproduced. Use directives to require multi-seed reruns for any "champion" candidate.
- **Tokenizer / vocab swaps that change BPB units.** A different tokenizer changes what one byte means in token space — comparing BPB across tokenizers is invalid. Annotate experiments using non-standard tokenizers as `control` or `exploration`, never `champion`.
