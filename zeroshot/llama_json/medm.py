import json
from calls import llama_call
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import chia
from constants import OUTPUT_DIR

def main():
    all_responses = {}
    prompt = chia.TEXT_JSON
    dataset = load_dataset("bigbio/medmentions", split="test")
    count = 0
    for item in tqdm(dataset):
        iid = item["pmid"]
        if item["passages"][0]["type"] == "title":
            title = item["passages"][0]["text"][0]
        abstract = item["passages"][1]["text"][0]
        text = title + " " + abstract
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)

        sent_response = {}
        for idx, this_text in enumerate(sent_text):
            messages = prompt + this_text
            response = llama_call.generate_text(messages, temperature=0, max_tokens=256
            )
            
            sent_response[idx] = response

        all_responses[iid] = sent_response
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            with open(
                OUTPUT_DIR / f"/final_zs/medm/llama/zs_text_json_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
