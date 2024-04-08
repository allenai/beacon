import utils
import json
import os
from calls import llama_call
import random
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import cdr
from constants import OUTPUT_DIR

HUMAN_PROMPT = "<human>:"
AI_PROMPT = "<bot>:"

def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = cdr.TEXT_JSON
    dataset = load_dataset("bigbio/bc5cdr")["test"]
    output_formatter = utils.OutputFormatter(dataset="cdr", seed=seed)
    count = 0
    for item in tqdm(dataset):
        iid = item["passages"][0]["document_id"]
        if item["passages"][0]["type"] == "title":
            title = item["passages"][0]["text"]
        abstract = item["passages"][1]["text"]
        text = title + " " + abstract
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)
        # sanity check if number of sentences == saved json sentence length
        sent_response = {}
        for sent_id, this_text in enumerate(sent_text):
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            #messages 
            messages = [prompt]
            for i in range(len(formatted_shots)):
                messages.append(
                    HUMAN_PROMPT
                    + " "
                    + formatted_shots[i]["text"]
                    + AI_PROMPT
                    + str(
                        json.dumps(
                            {
                                k: v
                                for k, v in formatted_shots[i].items()
                                if k in ("chemicals", "diseases")
                            }
                        ),
                    )
                )
            # sent_messages = str(messages) + (HUMAN_PROMPT + "\nSentence: " + text)
            messages.append(HUMAN_PROMPT + "\nSentence: " + this_text)
            messages = str(messages)
            response = llama_call.generate_text(messages, temperature=0, max_tokens=256)
            # tries = 1
            # while tries < 10:
            #     while claude_call.jsonNotFormattedCorrectly(response):
            #         user = "Please correct the json as it has syntax issues."
            #         correct_input = user + messages + response
            #         # edit how to send the old response
            #         # what should be the input here?
            #         response = claude_call.generate(correct_input)
            #         tries += 1
            sent_response[sent_id] = response

        all_responses[iid] = sent_response

        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_fs/cdr/llama/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs/cdr/llama/fs_text_json_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
