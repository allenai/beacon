import os
import json
import random
import utils
from calls import llama_call
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from utils import timeit
from prompts import medm
from constants import OUTPUT_DIR

HUMAN_PROMPT = "<human>:"
AI_PROMPT = "<bot>:"

@timeit("run_fewshot")
def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = medm.TEXT_JSON
    dataset = load_dataset("bigbio/medmentions", split="test")
    output_formatter = utils.OutputFormatter(dataset="medm", seed=seed)

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

        for sent_id, this_text in enumerate(sent_text):
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            messages += [prompt]
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
                                if k in ("entities")
                            }
                        ),
                    )
                )

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
            path = OUTPUT_DIR / "/final_fs/medm/llama/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs/medm/llama/zs_text_json_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
