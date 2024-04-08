import json
from calls import llama_call
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from prompts import chia
from constants import OUTPUT_DIR


def main():
    all_responses = {}
    prompt = chia.TEXT_JSON
    dataset = load_from_disk(
        ""replace with the chia data split path"/test.json"
    )

    count = 0
    for item in tqdm(dataset):
        iid = item["id"]
        all_text = item["text"]
        sent_text = filter(None, all_text.split("\n"))
        sent_response = {}
        if item["id"][-3:] == "exc":
            criterion = "\nGiven the above Exclution criterion for clinical trial, "
        else:
            criterion = "\nGiven the above Inclusion criterion for clinical trial, "
        for idx, text in enumerate(sent_text):
            messages = criterion +  prompt + "\n Sentence: " + text 
            response = llama_call.generate_text(
                messages, temperature=0, max_tokens=256
            )
            
            sent_response[idx] = response


        all_responses[iid] = sent_response
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            with open(
                OUTPUT_DIR / f"/final_zs/chia/llama/zs_def_json_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
