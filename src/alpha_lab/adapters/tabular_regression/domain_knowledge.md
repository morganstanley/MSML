# Domain Knowledge

Tabular regression: predict a continuous target from tabular features. Minimize mean squared error (MSE).

## Budget

Every model call (fit + predict) consumes budget. Check budget status with `python task/runner.py --budget`. Do not waste budget on redundant or exploratory model calls — read the spec first (`--spec`), plan your calls, and minimize the number of runs needed to achieve your goal.
