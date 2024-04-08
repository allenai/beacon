import json
import random
import utils
import os
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from calls import openai_call
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import cdr
from constants import OUTPUT_DIR

schema = {
    "type": "object",
    "properties": {
        "chemicals": {
            "type": "array",
            "description": "List of all the chemicals.",
            "items": {"type": "string"},
        },
        "diseases": {
            "type": "array",
            "description": "List of all the diseases.",
            "items": {"type": "string"},
        },
    },
    "required": ["chemicals", "diseases"],
}


def main(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = cdr.GPT_TEXT
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
        for idx, text in enumerate(sent_text):
            sent_id = idx
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            #print("check2")
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
                                if k in ("chemicals", "diseases")
                            }
                        ),
                    },
                ]

            messages += [{"role": "user", "content": text}]

            response = openai_call.generate_text(
                messages, schema, temperature=0, max_tokens=128
            )
            
            #print("check3")
            sent_response[sent_id] = response
            

        all_responses[iid] = sent_response
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_fs/cdr/gpt/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs/cdr/gpt/fs_text_json_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


# def main():
#     pass


if __name__ == "__main__":
    main(3, 12)
