import json
from calls import claude_call
from tqdm import tqdm
from datasets import load_from_disk
from prompts import chia
from constants import OUTPUT_DIR

def main():
    all_responses = {}
    prompt = chia.TEXT_JSON
    dataset = load_from_disk(
        "add the path "
        )
    count = 0
    for item in tqdm(dataset):
        iid = item["id"]
        all_text = item["text"]
        sent_text = filter(None, all_text.split("\n"))
        sent_response = {}
        if item["id"][-3:] == "exc":
            criterion = "\nGiven the above Exclution criterion for clinical trial, "
        else:
            criterion = "\nGiven the above Inclusion criterion for clinical trial, "
        for idx, text in enumerate(sent_text):
            messages = "Sentence: " + text + criterion + prompt
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
            print(f"Num of test datapoints: {count}")
            with open(
                OUTPUT_DIR / f"final_zs/cdr/claude2/zs_text_json_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
