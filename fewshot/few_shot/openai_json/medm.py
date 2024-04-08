import openai
import os
import json
import random
import utils
from calls import openai_call
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from utils import timeit
from prompts import medm
from constants import OUTPUT_DIR

openai.api_key = os.getenv("OPENAI_API_KEY")

schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["entities"],
}


@timeit("run_fewshot")
def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = medm.GPT_TEXT
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

        for idx, text in enumerate(sent_text):
            sent_id = idx
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            messages += [{"role": "user", "content": prompt}]

            for i in range(len(formatted_shots)):
                messages += [
                    {"role": "user", "content": formatted_shots[i]["text"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                k: v
                                for k, v in formatted_shots[i].items()
                                if k in ("entities")
                            }
                        ),
                    },
                ]

            messages += [{"role": "user", "content": text}]

            with timeit("openai_call"):
                response = openai_call.generate_text(
                    messages, schema, temperature=0, max_tokens=512
                )
                sent_response[sent_id] = response
           
        all_responses[iid] = sent_response
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_fs_subsample/medm/gpt4/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs_subsample/medm/gpt4/fs_text_json_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
