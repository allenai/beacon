import os
import json
import openai
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from calls import openai_call
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import cdr
from constants import OUTPUT_DIR

openai.api_key = os.getenv("OPENAI_API_KEY")

schema = {
    "type": "object",
    "properties": {
        "chemicals": {
            "type": "array",
            "description": "List of all the chemicals.",
            "items": {"type": "string"},
        },
        "diseases": {
            "type": "array",
            "description": "List of all the diseases.",
            "items": {"type": "string"},
        },
    },
    "required": ["chemicals", "diseases"],
}


def main():
    
    all_responses = {}
    prompt = cdr.GPT_TEXT   
    dataset = load_dataset("bigbio/bc5cdr")["test"]
    count = 0

    for item in tqdm(dataset):
        iid = item["passages"][0]["document_id"]
        if item["passages"][0]["type"] == "title":
            title = item["passages"][0]["text"]
        abstract = item["passages"][1]["text"]
        # rerunnning with space and the better tokenizer
        text = title + " " + abstract
        sent_text = []
        #sent_text = nltk.sent_tokenize(text)
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)

        sent_response = {}
        for sent_id, this_text in enumerate(sent_text):
            
            user = prompt + this_text + "\nOutput:"
            system = " "
            user1 = " "
            response = openai_call.generate_text(
                user, system, user1, schema, temperature=0, max_tokens=256
            )
            tries = 1
            while openai_call.jsonNotFormattedCorrectly(response):
                user = user
                system = response
                user1 = "Please correct the json it has syntax issues"
                response = openai_call.generate_text(
                    user, system, user1, temperature=0, max_tokens=256
                )
                tries += 1

            sent_response[sent_id] = response
 
        all_responses[iid] = sent_response
        # time.sleep(3)
        count += 1

        if count == len(dataset):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs/cdr/gpt4/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_zs/cdr/gpt4/zs_text_json.json",
                "w",
            ) as f:
                json.dump(all_responses, f)

            print("Saved")


if __name__ == "__main__":
    main()
