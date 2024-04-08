import cdr, chemprot, chia, pico, medm, ncbi

datasets = [ncbi, chemprot]
k_shots = [5]
seeds = [23, 42]

for dataset in datasets:
    for seed in seeds:
        for k in k_shots:
            dataset.run_fewshot(k, seed)


#chemprot- seeds 23 and 42 left