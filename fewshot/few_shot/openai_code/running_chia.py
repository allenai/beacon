import chia

k_shots = [1, 3, 5, 10, 15]
seeds = [12, 23, 42]

for seed in seeds:
    for k in k_shots:
        chia.run_fewshot(k, seed)
