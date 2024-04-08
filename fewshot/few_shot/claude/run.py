import chia
import pico


datasets = [pico]
k_shots = [5]
seeds = [23, 42]
# 23 and 42 if this works

for dataset in datasets:
    for seed in seeds:
        for k in k_shots:
            dataset.run_fewshot(k, seed)
