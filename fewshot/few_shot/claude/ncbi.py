import os
import json
import random
import utils
from calls import claude_call
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from prompts import ncbi
from constants import OUTPUT_DIR

def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = ncbi.TEXT_JSON
    dataset = load_dataset("bigbio/ncbi_disease", split="test")
    output_formatter = utils.OutputFormatter(dataset="ncbi", seed=seed)

    count = 0
    for item in tqdm(dataset):
        iid = item["pmid"]
        title = item["title"]
        abstract = item["abstract"]
        text = title + " " + abstract
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)
        sent_response = {}
        for idx, text in enumerate(sent_text):
            sent_id = idx
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
                                if k in ("diseases")
                            }
                        ),
                    )
                )
            # sent_messages = str(messages) + (HUMAN_PROMPT + "\nSentence: " + text)
            messages.append(HUMAN_PROMPT + "\nSentence: " + text)
            response = claude_call.generate(messages)

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
            path = OUTPUT_DIR / "/final_fs/ncbi/claude/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs/ncbi/claude/zs_text_json_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
