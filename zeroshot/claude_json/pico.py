import json
from calls import claude_call
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
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
        for idx, text in enumerate(sent_text):
            messages = "Sentence: " + text + prompt + "\nOutput:"
            response = claude_call.generate(messages)
            tries = 1
            while claude_call.jsonNotFormattedCorrectly(response):
                user = "Please correct the json as it has syntax issues."
                correct_input = user + messages + response
                # what should be the input here?
                response = claude_call.generate(correct_input)
                tries += 1
                print(tries)
            sent_response[idx] = response
        all_responses[iid] = sent_response
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            with open(
                OUTPUT_DIR / f"final_zs/cdr/claude2/zs_text_json_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
