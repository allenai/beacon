import json
import random
import utils
from calls import openai_call
import pandas as pd
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import pico
from constants import OUTPUT_DIR

entities_map = {
    "population": "population",
    "intervention": "intervention",
    "comparator": "comparator",
    "outcome": "outcome",
}


def extract_entities(input_dict):
    extracted_entities = []
    input_dict = dict(input_dict)
    # Extract PICO

    extracted_entities = []

    for key, entity_type in entities_map.items():
        entities = input_dict.get(key, [])
        for entity in entities:
            extracted_entities.append(
                f"entity_list.append({{'text': '{entity}', 'type': '{entity_type}'}})\n"
            )

    return "".join(extracted_entities)


def main(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = pico.CODE_TEXT
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
        for idx, text in enumerate(sent_text):
            sent_id = idx
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)

            messages = prompt
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
                + """\nentity_list = [] \n # extracted entities \n"""
            )

            messages += stuff

            response = openai_call.generate_text(
                str(messages), temperature=0, max_tokens=512
            )

            sent_response[sent_id] = response
            

        all_responses[iid] = sent_response
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            with open(
                OUTPUT_DIR / f"/final_fs/pico/gpt/zs_text_code_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)



if __name__ == "__main__":
    main(5, 12)
