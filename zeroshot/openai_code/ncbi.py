import json
from calls import openai_call
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import ncbi
from constants import OUTPUT_DIR

def main():
    all_responses = {}
    prompt = ncbi.CODE_TEXT
    dataset = load_dataset("bigbio/ncbi_disease")["test"]
    count = 0
    for item in tqdm(dataset):
        iid = item["pmid"]
        title = item["title"]
        abstract = item["abstract"]
        all_text = title + " " + abstract
        sent_text = []
        #sent_text = nltk.sent_tokenize(text)
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)

        sent_response = {}
        for idx, text in enumerate(sent_text):
            sent_id = idx
            user_input = (
                prompt
                + text
                + "\nentity_list = [] \n # extracted entities \n entity_list.append({'"
            )
            response = openai_call.generate_text(user_input, temperature=0, max_tokens=128)
            sent_response[sent_id] = response
            
        all_responses[iid] = sent_response
        # time.sleep(3)
        count += 1
        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            with open(
                OUTPUT_DIR / f"final_zs/ncbi/gpt/zs_text_code_test_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
