import openai
import os
import json
from calls import openai_call
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import medm
from constants import OUTPUT_DIR

openai.api_key = os.getenv("OPENAI_API_KEY")

schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["entities"],
}



def main():
    all_responses = {}
    prompt = medm.GPT_TEXT
    dataset = load_dataset("bigbio/medmentions", split="test")
    count = 0

    for item in tqdm(dataset):
        iid = item["pmid"]
        if item["passages"][0]["type"] == "title":
            title = item["passages"][0]["text"][0]
        abstract = item["passages"][1]["text"][0]
        text = title + abstract
        #sent_text = nltk.sent_tokenize(text)
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
                    user, system, user1, temperature=0, max_tokens=128
                )
                tries += 1

            sent_response[sent_id] = response

        all_responses[iid] = sent_response
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs/medm/gpt4/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_zs/medm/gpt4/zs_text_json.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
