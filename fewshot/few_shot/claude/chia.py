import json
import random
import utils
import os
from calls import claude_call
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_from_disk
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from prompts import chia
from constants import OUTPUT_DIR

entities_map = {
    "condition": "list of entities",
    "device": "list of entities",
    "drug": "list of entities",
    "measurement": "list of entities",
    "mood": "list of entities",
    "multiplier": "list of entities",
    "negation": "list of entities",
    "observation": "list of entities",
    "person": "list of entities",
    "procedure": "list of entities",
    "qualifier": "list of entities",
    "reference_point": "list of entities",
    "scope": "list of entities",
    "temporal": "list of entities",
    "value": "list of entities",
    "visit": "list of entities",
}


def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = chia.TEXT_JSON
    dataset = load_from_disk(
        ""replace with the chia data split path"/test.json"
    )
    print(len(dataset))
    output_formatter = utils.OutputFormatter(dataset="chia", seed=seed)
    count = 0
    for item in tqdm(dataset):
        iid = item["id"]
        all_text = item["text"]
        sent_text = filter(None, all_text.split("\n"))

        sent_response = {}
        if item["id"][-3:] == "exc":
            criterion = """
                Given the definitions of entities and following Exclution criterion for clinical trail, """
        else:
            criterion = """
                Given the following Inclusion criterion for clinical trail, """
        for idx, text in enumerate(sent_text):
            sent_id = idx
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            messages += [criterion + prompt]
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
                                if k in ("condition",
                                        "device",
                                        "drug",
                                        "measurement",
                                        "mood",
                                        "multiplier",
                                        "negation",
                                        "observation",
                                        "person",
                                        "procedure",
                                        "qualifier",
                                        "reference_point",
                                        "scope",
                                        "temporal",
                                        "value",
                                        "visit",)
                            }
                        ),
                    )
                )
            # sent_messages = str(messages) + (HUMAN_PROMPT + "\nSentence: " + text)
            messages.append(HUMAN_PROMPT + "\nSentence: " + text)
            response = claude_call.generate(messages)
            sent_response[sent_id] = response

        all_responses[iid] = sent_response
        count += 1

    if (count % 100 == 0) or (count == len(dataset)):
        print(f"Num of test datapoints: {count}")
        path = OUTPUT_DIR / "/final_fs/chia/claude/"
        isExist = os.path.exists(path)
        if not isExist:
            print("creating path")
            os.makedirs(path)
        with open(
            OUTPUT_DIR / f"/final_fs/chia/claude/fs_text_json_{k}shot_seed{seed}_{count}.json",
            "w",
        ) as f:
            json.dump(all_responses, f)


def main():
    pass


if __name__ == "__main__":
    main()
