import json
import random
import utils
import os
from calls import openai_call
import pandas as pd
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_from_disk
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from prompts import chia
from constants import OUTPUT_DIR

entities_map = {
    "condition": "list of entities",
    "device": "list of entities",
    "drug": "list of entities",
    "measurement": "list of entities",
    "mood": "list of entities",
    "multiplier": "list of entities",
    "negation": "list of entities",
    "observation": "list of entities",
    "person": "list of entities",
    "procedure": "list of entities",
    "qualifier": "list of entities",
    "reference_point": "list of entities",
    "scope": "list of entities",
    "temporal": "list of entities",
    "value": "list of entities",
    "visit": "list of entities",
}


schema = {
    "type": "object",
    "properties": {
        "Value": {
            "type": "array",
            "description": "List of all the values.",
            "items": {"type": "string"},
        },
        "Reference_points": {
            "type": "array",
            "description": "List of all the reference points.",
            "items": {"type": "string"},
        },
        "Devices": {
            "type": "array",
            "description": "List of all the devices.",
            "items": {"type": "string"},
        },
        "Multiplier": {
            "type": "array",
            "description": "List of all the multipliers.",
            "items": {"type": "string"},
        },
        "Condition": {
            "type": "array",
            "description": "List of all the conditions.",
            "items": {"type": "string"},
        },
        "Temporal": {
            "type": "array",
            "description": "List of all the temporal info.",
            "items": {"type": "string"},
        },
        "Person": {
            "type": "array",
            "description": "List of all the Person.",
            "items": {"type": "string"},
        },
        "Drug": {
            "type": "array",
            "description": "List of all the Drugs.",
            "items": {"type": "string"},
        },
        "Negation": {
            "type": "array",
            "description": "List of all the Negations.",
            "items": {"type": "string"},
        },
        "Measurement": {
            "type": "array",
            "description": "List of all the Measurements.",
            "items": {"type": "string"},
        },
        "Procedure": {
            "type": "array",
            "description": "List of all the Procedures.",
            "items": {"type": "string"},
        },
        "Visit": {
            "type": "array",
            "description": "List of all the Visits.",
            "items": {"type": "string"},
        },
        "Mood": {
            "type": "array",
            "description": "List of all the Moods.",
            "items": {"type": "string"},
        },
        "Qualifier": {
            "type": "array",
            "description": "List of all the Qualifiers.",
            "items": {"type": "string"},
        },
        "Observation": {
            "type": "array",
            "description": "List of all the Observations.",
            "items": {"type": "string"},
        },
        "Scope": {
            "type": "array",
            "description": "List of all the Scopes.",
            "items": {"type": "string"},
        },
    },
    "required": [
        "Value",
        "Reference_point",
        "Device",
        "Multiplier",
        "Condition",
        "Temporal",
        "Person",
        "Drug",
        "Negation",
        "Measurement",
        "Procedure",
        "Visit",
        "Mood",
        "Qualifier",
        "Observation",
        "Scope",
    ],
}



def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = chia.GPT_DEF

    # dataset = load_from_disk(
    #     ""replace with the chia data split path"/test.json"
    # )

    dataset = load_from_disk(
        "replace with the chia data split path"
    )
    print(len(dataset))
    output_formatter = utils.OutputFormatter(dataset="chia", seed=seed)
    count = 0
    for item in tqdm(dataset):
        iid = item["id"]
        all_text = item["text"]
        sent_text = filter(None, all_text.split("\n"))

        sent_response = {}
        if item["id"][-3:] == "exc":
            criterion = """
                Given the definitions of entities and following Exclution criterion for clinical trail, """
        else:
            criterion = """
                Given the following Inclusion criterion for clinical trail, """
        for idx, text in enumerate(sent_text):
            sent_id = idx
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            messages += [{"role": "user", "content": criterion + prompt}]
            for i in range(len(formatted_shots)):
                messages += [
                    {"role": "user", "content": formatted_shots[i]["text"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                k: v
                                for k, v in formatted_shots[i].items()
                                if k in ("condition",
                                        "device",
                                        "drug",
                                        "measurement",
                                        "mood",
                                        "multiplier",
                                        "negation",
                                        "observation",
                                        "person",
                                        "procedure",
                                        "qualifier",
                                        "reference_point",
                                        "scope",
                                        "temporal",
                                        "value",
                                        "visit",)
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
        path = OUTPUT_DIR / "/final_fs_subsample/chia/gpt4/"
        isExist = os.path.exists(path)
        if not isExist:
            print("creating path")
            os.makedirs(path)
        with open(
            OUTPUT_DIR / f"/final_fs_subsample/chia/gpt4/fs_text_json_{k}shot_seed{seed}_{count}.json",
            "w",
        ) as f:
            json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
