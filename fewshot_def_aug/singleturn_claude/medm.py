import openai
import os
import json
import nltk
import random
import ast
import utils
from calls import claude_call
import pandas as pd
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from utils import timeit
from calls import retrieval
from prompts import common, medm
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


def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = medm.TEXT_JSON
    dataset = load_dataset("bigbio/medmentions")["test"]

    #dataset = load_dataset("bigbio/medmentions", split="test")
    output = OUTPUT_DIR / f"/final_fs_subsample/medm/claude/zs_text_json_5shot_seed{seed}_100.json"
    extracted_entities = json.load(open(output))
    output_formatter = utils.OutputFormatter(dataset="medm", seed=seed)
    retriever = retrieval.KnowledgeRetrieval()
    count = 0
    keyerr = 0 
    synerr = 0
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

        extracted_abs = extracted_entities[iid]
        sent_response = {}

        for sent_id, this_text in enumerate(sent_text):

            try:
                extracted_sent = extracted_abs[str(sent_id)]
                #extracted_sent = ast.literal_eval(extracted_sent_raw.strip())
            except SyntaxError as e:
                synerr += 1
                extracted_sent = {"entity" : []}

            except KeyError as e:
                keyerr += 1
                extracted_sent = {"entity" : []}


            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            messages += [prompt]

            for i in range(len(formatted_shots)):

                shot_text = formatted_shots[i]["text"]
                messages += [HUMAN_PROMPT + " " +  shot_text]
                list_to_retrieve = []
                #check what this is?
                list_to_retrieve += formatted_shots[i]["entities"] 
                
                noun_definitions = retriever.extract_noun_phrases_and_link_with_umls_remove_repeats(shot_text, list_to_retrieve)
                ent_definitions = retriever.link_with_umls(list_to_retrieve)

                content = ""
                for key, value in ent_definitions.items():
                    content += f"{key}: {value}\n"

                content_noun = ""
                for key, value in noun_definitions.items():
                    content_noun += f"{key}: {value}\n"

                if content == "":
                    if content_noun == "":
                        messages += [HUMAN_PROMPT +  common.PROMPT_END_NO_DEF + "\nOutput:" ]
                    else: 
                        messages += [HUMAN_PROMPT + common.PROMPT_ENT + content_noun +  common.PROMPT_END + "\n Sentence: " + this_text + "\nOutput:" ]
                else:
                    if content_noun == "":
                        messages += [HUMAN_PROMPT + common.PROMPT_ENT + content +  common.PROMPT_END + "\n Sentence: " + this_text + "\nOutput:" ]
                    else:   
                        messages += [HUMAN_PROMPT + common.PROMPT_ENT + content + content_noun +  common.PROMPT_ENT + "\n Sentence: " + this_text + "\nOutput:" ]
    

                messages += (str(json.dumps(
                                {
                                    k: v
                                    for k, v in formatted_shots[i].items()
                                    if k in ("entities")
                                }
                            ),
                        ),
                )

            messages += [HUMAN_PROMPT + " " + this_text]
            
            list_to_retrieve_text = []
            for types, entities in extracted_sent.items():
                list_to_retrieve_text += entities
            
            noun_definitions = retriever.extract_noun_phrases_and_link_with_umls_remove_repeats(this_text, list_to_retrieve_text)
            ent_definitions = retriever.link_with_umls(list_to_retrieve_text)

            content = ""
            for key, value in ent_definitions.items():
                content += f"{key}: {value}\n"

            content_noun = ""
            for key, value in noun_definitions.items():
                content_noun += f"{key}: {value}\n"

            if content == "":
                if content_noun == "":
                    messages += [HUMAN_PROMPT +  common.PROMPT_END_NO_DEF + "\nOutput:" ]
                else: 
                    messages += [HUMAN_PROMPT + common.PROMPT_ENT + content_noun +  common.PROMPT_END+ "\n Sentence: " + this_text + "\nOutput:" ]
            else:
                if content_noun == "":
                    messages += [HUMAN_PROMPT + common.PROMPT_ENT + content +  common.PROMPT_END+ "\n Sentence: " + this_text + "\nOutput:" ]
                else:   
                    messages += [HUMAN_PROMPT + common.PROMPT_ENT + content + content_noun +  common.PROMPT_END+ "\n Sentence: " + this_text + "\nOutput:" ]
            
            response = claude_call.generate(messages)

            sent_response[sent_id] = response

        all_responses[iid] = sent_response
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_fs_ret/medm/claude/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs_ret/medm/claude/fs_stC_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)

    print(keyerr)
    print(synerr)

def main():
    pass


if __name__ == "__main__":
    main()
