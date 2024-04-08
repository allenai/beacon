import json
from calls import llama_call
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import pico
from constants import OUTPUT_DIR

def main():
    all_responses = {}
    prompt = pico.TEXT_JSON
    dataset = load_dataset("bigbio/ebm_pico", split="test")
    count = 0

    for item in tqdm(dataset):
        text = item["text"]
        iid = item["doc_id"]
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)
        sent_response = {}
        for idx, this_text in enumerate(sent_text):
            messages = prompt + "\n Sentence: " + this_text 
            response = llama_call.generate_text(
                messages, temperature=0, max_tokens=256
            )

            sent_response[idx] = response

        all_responses[iid] = sent_response
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            with open(
                OUTPUT_DIR / f"/final_zs/pico/llama/zs_text_json_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
