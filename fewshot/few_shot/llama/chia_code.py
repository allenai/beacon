import json
import random
import utils
import os
from calls import llama_call
import pandas as pd
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_from_disk
from prompts import chia
from constants import OUTPUT_DIR

HUMAN_PROMPT = "<human>:"
AI_PROMPT = "<bot>:"

entities_map = {
    "condition": "condition",
    "device": "device",
    "drug": "drug",
    "measurement": "measurement",
    "mood": "mood",
    "multiplier": "multiplier",
    "negation": "negation",
    "observation": "observation",
    "person": "person",
    "procedure": "procedure",
    "qualifier": "qualifier",
    "reference_point": "reference_point",
    "scope": "scope",
    "temporal": "temporal",
    "value": "value",
    "visit": "visit",
}


def extract_entities(input_dict):
    extracted_entities = []
    input_dict = dict(input_dict)
    extracted_entities = []

    for key, entity_type in entities_map.items():
        entities = input_dict.get(key, [])
        for entity in entities:
            extracted_entities.append(
                f"entity_list.append({{'text': '{entity}', 'type': '{entity_type}'}})\n"
            )

    return "".join(extracted_entities)


def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = chia.CODE_TEXT
    dataset = load_from_disk(
        ""replace with the chia data split path"/test.json"
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
                def named_entity_recognition(input_text):
                \""" Given the definitions of entities and following Exclution criterion for clinical trail, """
        else:
            criterion = """
                def named_entity_recognition(input_text):
                \""" Given the following Inclusion criterion for clinical trail, """
        for idx, text in enumerate(sent_text):
            sent_id = idx
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)

            messages = criterion + prompt
            for i in range(len(formatted_shots)):
                result = extract_entities(formatted_shots[i])
                if result:
                    k_input = (
                        "\ninput_text = "
                        + formatted_shots[i]["text"]
                        + """\nentity_list = [] \n # extracted entities \n """
                        + str(result)
                    )
                else:
                    k_input = (
                        "\ninput_text = "
                        + formatted_shots[i]["text"]
                        + """\nentity_list = [] \n # extracted entities \n """
                        + "entity_list=[]"
                    )
                messages += k_input

            stuff = (
                "input_text = "
                + text
                #+ """\nentity_list = [] \n # extracted entities \n"""
                + "\nentity_list = [] \n # extracted entities \n entity_list.append({'"
            )

            messages += stuff
            response = llama_call.generate_text(messages, temperature=0, max_tokens=512)
            sent_response[sent_id] = response

        all_responses[iid] = sent_response
        count += 1

    if (count % 100 == 0) or (count == len(dataset)):
        print(f"Num of test datapoints: {count}")
        path = OUTPUT_DIR / "/final_fs/chia/llama/"
        isExist = os.path.exists(path)
        if not isExist:
            print("creating path")
            os.makedirs(path)
        with open(
            OUTPUT_DIR / f"/final_fs/chia/llama/fs_text_code_{k}shot_seed{seed}_{count}.json",
            "w",
        ) as f:
            json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
