import cdr, chemprot, chia, pico, medm, ncbi

datasets = [cdr, chemprot, chia, pico, medm, ncbi]
k_shots = [5]
seeds = [12, 23, 42]

for dataset in datasets:
    for seed in seeds:
        for k in k_shots:
            dataset.run_fewshot(k, seed)
