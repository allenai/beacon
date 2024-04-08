import os
from ipdb import set_trace
import eval_json, eval_code
import numpy as np
import traceback
from constants import OUTPUT_DIR

"""
This is to evaluate fewshot with full test set
"""
DATASET_EVAL_SCRIPT_MAP = {
    "cdr": {"json": eval_json.cdr, "code": eval_code.cdr},
    "chemprot": {"json": eval_json.chemprot, "code": eval_code.chemprot},
    "chia": {"json": eval_json.chia, "code": eval_code.chia},
    "medm": {"json": eval_json.medm, "code": eval_code.medm},
    "ncbi": {"json": eval_json.ncbi, "code": eval_code.ncbi},
    "pico": {"json": eval_json.pico, "code": eval_code.pico}
}

EVAL_DIR = OUTPUT_DIR / f"final_fs/"
FILE_FORMAT = "fs_text_{kind}_{k}shot_seed{seed}"
FILE_TYPES_TO_EVAL = ["json", "code"]

K_SHOTS = [1, 3, 5]
SEEDS = [12, 23, 42]
#instead of input types, there are seeds and k shots

ENTITY_TYPES_MAP = {"cdr": 500 , "chemprot": 800, "medm": 879, "ncbi" : 100, "pico": 187, "chia": 600}

def _get_all_data_files(dataset_dir):
    return os.listdir(os.path.join(EVAL_DIR, dataset_dir))

def _get_file_to_eval(all_files, kind, k_shot, seed):
    filtered_files = []
    num_datapoints = []

    file_filter = FILE_FORMAT.format(kind=kind, k=k_shot, seed=seed)
    
    for file in all_files:
        filename, ext = os.path.splitext(file)
        if ext == ".json" and filename.startswith(file_filter):
            filtered_files.append(file)

            try:
                num_datapoints.append(int(filename.split("_")[-1]))
            except ValueError:
                return None


    if not filtered_files:
        return None

    if not num_datapoints and len(filtered_files) == 1:
        return filtered_files[0]

    try:
        idx_to_select = np.argmax(num_datapoints)
    except ValueError:
        set_trace()
    except UnboundLocalError:
        set_trace()

    return filtered_files[idx_to_select]


def main(dry_run=True):
    datasets_to_eval = os.listdir(EVAL_DIR)
    output_file = open("output.txt", "w")
    errors = []

    for dataset in datasets_to_eval:
        
        dataset_abs_path = os.path.join(EVAL_DIR, dataset)
        for model_type in os.listdir(dataset_abs_path):
            dataset_model_type_abs_path = os.path.join(dataset_abs_path, model_type)
            for file_type in FILE_TYPES_TO_EVAL:
                for k_shot in K_SHOTS:
                    f1_seeds = []
                    for seed in SEEDS:
                        print("-" * 100)
                        print(f"Running evaluation for: {dataset}/{model_type}/{file_type}/{k_shot}/{seed}")
                        print("\n\n")

                        eval_file = _get_file_to_eval(os.listdir(dataset_model_type_abs_path), kind=file_type, k_shot=k_shot, seed=seed)
                        if not eval_file:
                            continue

                        run_eval = DATASET_EVAL_SCRIPT_MAP[dataset][file_type]

                        if dry_run:
                            print(f"Script: {run_eval}")
                            print(f"Dataset: {os.path.join(dataset_model_type_abs_path, eval_file)}")
                            print(f"{eval_file}")
                            print()
                        else:
                            input_file = os.path.join(dataset_model_type_abs_path, eval_file)

                            print("-" * 100)
                            print(f"Running evaluation for: {dataset}/{model_type}/{file_type}/{k_shot}{seed}\n")
                            print("\n\n")
                            try:
                                #commented so it doesn't run and change files
                                # if dataset == "pico" or dataset == "chia":
                                #     precision, recall = run_eval.main(input_file)
                                # else:
                                if file_type == "code":
                                    precision, recall = run_eval.main(input_file)
                                else:
                                    precision, recall = run_eval.main(input_file)
                                f1_score = (2 * precision * recall) / (precision + recall)
                                f1_seeds.append(f1_score)
                                round_f1_score = round(f1_score, 4)
                            except Exception as exc:
                                errors.append(exc)
                                traceback.print_exception(type(exc), exc, exc.__traceback__)
                                continue

                            output_file.write(f"{dataset}/{model_type}/{file_type}/{k_shot}/{seed}\n")
                            output_file.write(f"File evaluated : {eval_file}\n")
                            output_file.write(f"Precision: {precision}\nRecall: {recall}\nF1: {round_f1_score}\n")

                            print(f"Precision: {precision}\nRecall: {recall}\nF1: {round_f1_score}")

                    if not dry_run:
                        if f1_seeds:
                            output_file.write(f"\nAvg F1: {round(sum(f1_seeds) / len(f1_seeds), 4)} \n")
                            output_file.write(f"\nStd Dev F1: {round(np.std(f1_seeds), 4)} \n\n\n")

                            print(f"\nAvg F1: {round(sum(f1_seeds) / len(f1_seeds), 4)}")
                            print(f"\nStd Dev F1: {round(np.std(f1_seeds), 4)}")

                    print("-" * 100)
                    print("\n\n")
                            
                
    output_file.close()
    print("\n\n\n\n")
    print("-"*100)
    print("ERRORS\n\n")
    for ex in errors:
        traceback.print_exception(type(ex), ex, ex.__traceback__)


if __name__ == "__main__":
    main(dry_run=False)
