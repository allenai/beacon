import json
import random
import utils
import os
from calls import llama_call
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from anthropic import HUMAN_PROMPT, AI_PROMPT
from prompts import pico
from constants import OUTPUT_DIR

HUMAN_PROMPT = "<human>:"
AI_PROMPT = "<bot>:"

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


def run_fewshot(k, seed):
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
        for sent_id, text in enumerate(sent_text):
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
            path = OUTPUT_DIR / "/final_fs/pico/llama/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs/pico/llama/fs_text_code_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
