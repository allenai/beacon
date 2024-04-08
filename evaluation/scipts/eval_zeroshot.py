import os
from ipdb import set_trace
import eval_json, eval_code
import numpy as np
import traceback
from constants import OUTPUT_DIR

"""
This is to evaluate zeroshot with full test set
"""

DATASET_EVAL_SCRIPT_MAP = {
    "cdr": {"json": eval_json.cdr, "code": eval_code.cdr},
    "chemprot": {"json": eval_json.chemprot, "code": eval_code.chemprot},
    "chia": {"json": eval_json.chia, "code": eval_code.chia},
    "medm": {"json": eval_json.medm, "code": eval_code.medm},
    "ncbi": {"json": eval_json.ncbi, "code": eval_code.ncbi},
    "pico": {"json": eval_json.pico, "code": eval_code.pico}
}

EVAL_DIR = OUTPUT_DIR / f"final_zs/"
FILE_TYPES_TO_EVAL = ["code", "json"]
INPUT_TYPES_TO_EVAL = ["text", "def"]

ENTITY_TYPES_MAP = {"cdr": 500 , "chemprot": 800, "medm": 879, "ncbi" : 100, "pico": 187, "chia": 600}

def _get_all_data_files(dataset_dir):
    return os.listdir(os.path.join(EVAL_DIR, dataset_dir))

def _get_file_to_eval(all_files, kind, inp_type):
    filtered_files = []
    num_datapoints = []

    for file in all_files:
        filename, ext = os.path.splitext(file)
        if kind in filename:
            if ext == ".json" and inp_type in filename:
                filtered_files.append(file)

                try:
                    num_datapoints.append(int(filename.split("_")[-1]))
                except ValueError:
                    return None
                
            elif ext == ".json" and inp_type in filename:
                filtered_files.append(file)


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
    print(filtered_files[idx_to_select])
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
                for input_type in INPUT_TYPES_TO_EVAL:

                    eval_file = _get_file_to_eval(os.listdir(dataset_model_type_abs_path), kind=file_type, inp_type=input_type)
                    if not eval_file:
                        continue

                    run_eval = DATASET_EVAL_SCRIPT_MAP[dataset][file_type]

                    if dry_run:
                        print(f"Script: {run_eval}")
                        print(f"Dataset: {os.path.join(dataset_model_type_abs_path, eval_file)}")
                        print(f"Hardcoded entity: {ENTITY_TYPES_MAP[dataset]}")
                        print()
                    else:
                        input_file = os.path.join(dataset_model_type_abs_path, eval_file)

                        print("-" * 100)
                        print(f"Running evaluation for: {dataset}/{model_type}/{input_type}/{file_type}")
                        print("\n\n")

                        try:
                            precision, recall = run_eval.main(input_file)
                            f1_score = (2 * precision * recall) / (precision + recall)
                            round_f1_score = round(f1_score, 4)
                        except Exception as exc:
                            errors.append(exc)
                            traceback.print_exception(type(exc), exc, exc.__traceback__)
                            continue

                        output_file.write(f"{dataset}/{model_type}/{input_type}/{file_type}\n")
                        output_file.write(f"File evaluated : {eval_file}\n")
                        output_file.write(f"Precision: {precision}\nRecall: {recall}\nF1: {round_f1_score}\n\n\n")

                        print(f"Precision: {precision}\nRecall: {recall}\nF1: {round_f1_score}")
        

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
