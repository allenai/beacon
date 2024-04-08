import json
import random
import utils
import os
from calls import openai_call
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import pico
from constants import OUTPUT_DIR

schema = {
    "type": "object",
    "properties": {
        "population": {
            "type": "array",
            "description": "list of participants",
            "items": {"type": "string"},
        },
        "intervention": {
            "type": "array",
            "description": "list of all interventions",
            "items": {"type": "string"},
        },
        "comparator": {
            "type": "array",
            "description": "list of all the comparators mentioned in the study",
            "items": {"type": "string"},
        },
        "outcome": {
            "type": "array",
            "description": "outcome of the study",
            "items": {"type": "string"},
        },
    },
    "required": ["population", "intervention", "comparator", "outcome"],
}


def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = pico.GPT_TEXT
    dataset = load_dataset("bigbio/ebm_pico", split="test")
    output_formatter = utils.OutputFormatter(dataset="pico", seed=seed)
    count = 0
    for item in tqdm(dataset):
        text = item["text"]
        iid = item["doc_id"]
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)

        sent_response = {}
        for sent_id, text in enumerate(sent_text):
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            messages += [{"role": "user", "content": prompt}]
            for i in range(len(formatted_shots)):
                messages += [
                    {"role": "user", "content": formatted_shots[i]["text"]},
                    {
                        "role": "assistant",
                        "content":
                        json.dumps(
                            {
                                k: v
                                for k, v in formatted_shots[i].items()
                                if k in ("population", "intervention", "comparator", "outcome")
                            }
                        ),
                    },
                ]

            messages += [{"role": "user", "content": text}]

            response = openai_call.generate_text(
                messages, schema, temperature=0, max_tokens=128
            )

            sent_response[sent_id] = response

        all_responses[iid] = sent_response

        count += 1

    if (count % 100 == 0) or (count == len(dataset)):
        print(f"Num of test datapoints: {count}")
        path = OUTPUT_DIR / "/final_fs_subsample/pico/gpt/"
        isExist = os.path.exists(path)
        if not isExist:
            print("creating path")
            os.makedirs(path)
        with open(
            OUTPUT_DIR / f"/final_fs_subsample/pico/gpt/fs_text_json_{k}shot_seed{seed}_{count}.json",
            "w",
        ) as f:
            json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
