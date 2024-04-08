import json
from calls import llama_call
import os
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import cdr
from constants import OUTPUT_DIR

def main():
    all_responses = {}
    prompt = cdr.CODE_DEF
    dataset = load_dataset("bigbio/bc5cdr")["test"]
    count = 0
    for item in tqdm(dataset):
        iid = item["passages"][0]["document_id"]
        if item["passages"][0]["type"] == "title":
            title = item["passages"][0]["text"]
        abstract = item["passages"][1]["text"]
        text = title + " " + abstract
        sent_text = []
        #sent_text = nltk.sent_tokenize(text)
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)
        sent_response = {}
        for idx, this_text in enumerate(sent_text):
            user_input = (
                prompt
                + this_text
                + "\nentity_list = [] \n # extracted entities \n entity_list.append({'"
            )
            response = llama_call.generate_text(user_input, temperature=0, max_tokens=256)
            sent_response[idx] = response

        all_responses[iid] = sent_response
        count += 1

        if count == len(dataset):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs/cdr/llama/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_zs/cdr/llama/zs_text_code_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
