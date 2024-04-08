import pico
import chia


datasets = [pico]
k_shots = [5]
seeds = [12]

for dataset_mod in datasets:
    for seed in seeds:
        for k in k_shots:
            dataset_mod.run_fewshot(k, seed)
