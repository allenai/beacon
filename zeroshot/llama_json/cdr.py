import json
from tqdm import tqdm
from datasets import load_dataset
from calls import llama_call
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import cdr
from constants import OUTPUT_DIR

def main():
    
    all_responses = {}
    prompt = cdr.TEXT_JSON
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
            
            user_input = prompt + this_text
            response = llama_call.generate_text(
                user_input, temperature=0, max_tokens=256
            )

            sent_response[sent_id] = response


        all_responses[iid] = sent_response
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            with open(
                OUTPUT_DIR / f"/final_zs/cdr/llama/zs_text_json_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)

            print("Saved")


if __name__ == "__main__":
    main()
