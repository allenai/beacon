import json
import os
from calls import openai_call
from tqdm import tqdm
from datasets import load_from_disk
from prompts import chia
from constants import OUTPUT_DIR

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



def main():

    all_responses = {}
    prompt = chia.GPT_TEXT
    dataset = load_from_disk(
        ""replace with the chia data split path"/val.json"
    )
    count = 0
    for item in tqdm(dataset):
        iid = item["id"]
        all_text = item["text"]
        sent_text = filter(None, all_text.split("\n"))
        # sent_text = nltk.sent_tokenize(all_text)
        sent_response = {}
        # change criterion to calibration_prompt
        if item["id"][-3:] == "exc":
            criterion = "Given the Exclution criterion for a clinical trial "
        else:
            criterion = "Given the Inclusion criterion for a clinical trial "
        for idx, text in enumerate(sent_text):
            user = criterion + prompt + text + "\nOutput:"
            system = " "
            user1 = " "
            response = openai_call.generate_text(
                user, system, user1, schema, temperature=0, max_tokens=256
            )
            tries = 1
            try:
                while openai_call.jsonNotFormattedCorrectly(response):
                    user = user
                    system = response
                    user1 = "Please correct the json it has syntax issues"
                    response = openai_call.generate_text(
                        user, system, user1, temperature=0, max_tokens=256
                    )
                    tries += 1

            except ValueError as e:
                print(response)
                pass

            sent_response[idx] = response

        all_responses[iid] = sent_response
        count += 1
        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs/chia/gpt4/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_zs/chia/gpt4/zs_text_json.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
