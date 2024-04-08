import cdr, chemprot, chia, pico, medm, ncbi

datasets = [ncbi]
k_shots = [5]
seeds = [23]

for dataset in datasets:
    for seed in seeds:
        for k in k_shots:
            dataset.run_fewshot(k, seed)

# run chemprot seeds 23 and 42