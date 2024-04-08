import json
from calls import openai_call
import os
from tqdm import tqdm
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


def main():

    all_responses = {}
    prompt = pico.GPT_TEXT
    dataset = load_dataset("bigbio/ebm_pico", split="test")
    count = 0

    for item in tqdm(dataset):
        text = item["text"]
        iid = item["doc_id"]
        #sent_text = nltk.sent_tokenize(all_text)
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)
        sent_response = {}
        for idx, text in enumerate(sent_text):
            sent_id = idx
            user = prompt + text + "\nOutput:"
            system = " "
            user1 = " "
            response = openai_call.generate_text(
                user, system, user1, schema, temperature=0, max_tokens=128
            )
            tries = 1
            while openai_call.jsonNotFormattedCorrectly(response):
                user = user
                system = response
                user1 = "Please correct the json it has syntax issues"
                response = openai_call.generate_text(
                    user, system, user1, schema, temperature=0, max_tokens=128
                )
                tries += 1
            sent_response[sent_id] = response   

        all_responses[iid] = sent_response
        count += 1

    if (count % 100 == 0) or (count == len(dataset)):
        print(f"Num of test datapoints: {count}")
        path = OUTPUT_DIR / "/final_zs/pico/gpt4/"
        isExist = os.path.exists(path)
        if not isExist:
            print("creating path")
            os.makedirs(path)
        with open(
            OUTPUT_DIR / f"/final_zs/pico/gpt4/zs_text_json.json",
            "w",
        ) as f:
            json.dump(all_responses, f)


if __name__ == "__main__":
    main()
