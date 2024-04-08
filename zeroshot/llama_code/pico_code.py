import json
import os
from calls import llama_call
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import pico
from constants import OUTPUT_DIR

def main():
    all_responses = {}
    prompt = pico.CODE_TEXT
    dataset = load_dataset("bigbio/ebm_pico", split="test")
    count = 0

    for item in tqdm(dataset):
        text = item["text"]
        iid = item["doc_id"]
        sent_text = []
        #sent_text = nltk.sent_tokenize(text)
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)
        sent_response = {}
        for idx, text in enumerate(sent_text):
            user_input = (
                prompt
                + "input_test = "
                + text
                + "\nentity_list = [] \n # extracted entities \n entity_list.append({'"
            )
            response = llama_call.generate_text(user_input, temperature=0, max_tokens=256)
            sent_response[idx] = response
        all_responses[iid] = sent_response
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs/pico/llama/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_zs/pico/llama/zs_text_code_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
