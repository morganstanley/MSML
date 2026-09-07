# Domain Knowledge

Black-box optimization: find the global minimum of an unknown function f: [0,1]^d -> R.

## Interface

Each experiment proposes a single input x via a `strategy(X, y)` function. The blackbox module evaluates f(x) and enforces an evaluation budget.

## Strategy contract

Define `strategy(X, y)` in `strategy.py`:
- X: (n, d) array of previously queried points
- y: (n,) array of (noisy) observations
- Returns: (d,) array in [0, 1]^d

## Blackbox API

```python
import blackbox
from strategy import strategy

blackbox.smoke_test(strategy)   # raises ValueError if strategy is malformed, returns None on success
blackbox.evaluate(strategy)     # call strategy, evaluate f(x), write results/metrics.json
blackbox.create_runner()        # write a mechanical run_experiment.py in the current directory
```

## Noise

Observations may be corrupted by noise: y = f(x) + epsilon.

## Budget

The evaluation budget is finite. Once exhausted, `blackbox.evaluate(strategy)` raises a `RuntimeError`. STRICT: do not call `blackbox.evaluate(strategy)` directly. Use `blackbox.smoke_test(strategy)` to verify that a `strategy` function works.
