# hiriluk

## 1. Using hiriluk

Run from the repository root:

```bash
conda create -n token python=3.12 pip -y
conda activate token
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi
rustup show
python -m pip install --upgrade pip
python -m pip install -e . --config-settings=build-args=--locked
```

```python
import hiriluk

# The default uses the streaming DFA scanner and lookup cache.
chopper = hiriluk.get_chopper("r50k")
tokens = chopper.chop("Hello, world!")
print(tokens, tokens.dtype)  # NumPy uint32 array

# The Gigatoken-style SIMD scanner and full pretoken cache remain streaming
# for file input while returning the complete token array here.
fast = hiriluk.get_chopper("r50k", gigatoken=True)
tokens = fast.chop_file("input.txt")

# Rust can instead stream readable JSON directly to a file. The return value
# is the number of tokens written.
count = fast.chop_file(
    "input.txt",
    output="json",
    dump="tokens.json",
)
```

## 2. Benchmark setup

```bash
python -m pip install -e ".[benchmark,test]" --config-settings=build-args=--locked
python -m pip check
cargo build --release --locked
```

To download data:

```bash
# Edit DATA_DIR in paths.env to set download path
python tools/get_data.py
```

The data tool creates an approximately 16 MiB `tiny` corpus, an approximately
128 MiB default `short` corpus, and the complete `full` stream for each source.

To build the instrumented TTFT reference implementations:

```bash
# Set TTFT_REPOS_DIR in paths.env to a directory outside this repository.
rustup toolchain install nightly --profile minimal
python tools/setup_ttft_repos.py

# The setup command prints the exact `git -C ... diff` commands for inspection.
```

To run tests:

```bash
cargo test --locked --lib
python -m pytest python/tests
```

## 3. Running benchmarks

Throughput and TTFT experiments test every model/encoding against GitHub, 
English, and Chinese by default. Add `--tiny` for quick approximately 16
MiB runs, or selectors such as `--encoding r50k --dataset english` and
`--model gpt2 --dataset github`.

Run time-to-first-token experiments. The first command measures Hiriluk;
the external commands measure the patched reference implementations. The
external runners default to all four tiktoken encodings and all three default
corpora; their repetition counts can be overridden with `WARMUP` and `REPS`.

```bash
python benchmarks/tiktoken_ttft_bench.py --dfa
python benchmarks/hf_ttft_bench.py

python benchmarks/external/tiktoken_ttft_bench.py
python benchmarks/external/gigatoken_ttft_bench.py

# Example shorter external run configuration:
WARMUP=1 REPS=3 python benchmarks/external/gigatoken_ttft_bench.py
```

Run RSS experiments. 

```bash
python benchmarks/tiktoken_rss_bench.py --dfa
python benchmarks/hf_rss_bench.py

# Focus a case, change corpus size, or choose the artifact directory.
python benchmarks/tiktoken_rss_bench.py \
  --encoding cl100k --dataset github --full \
  --output-dir /tmp/hiriluk-rss
```

Run throughput comparisons:

```bash
# Compare tiktoken with Hiriluk's streaming DFA path.
python benchmarks/tiktoken_throughput_bench.py --dfa

# tiktoken encodings: tiktoken, serial Gigatoken, and fast path.
python benchmarks/tiktoken_throughput_bench.py

# Hugging Face Tokenizers versus Hiriluk's DFA path.
python benchmarks/hf_throughput_bench.py
```
