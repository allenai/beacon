import os
import json
import ast
import nltk
import random
from calls import llama_call
import pandas as pd
from tqdm import tqdm
from ipdb import set_trace
from collections import defaultdict
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from calls import retrieval
import wiki
from prompts import common, medm
from constants import OUTPUT_DIR


def main():
    all_responses = defaultdict(dict)
    prompt = medm.GPT_TEXT
    dataset = load_dataset("bigbio/medmentions")["test"]

    #run this again for retrieval
    output = OUTPUT_DIR / "final_zs_subsample/medm/llama/zs_text_json_100_for_retrieval.json"

    extracted_entities = json.load(open(output))
    linker = retrieval.KnowledgeRetrieval()
    retriever = wiki.KnowledgeRetrieval()
    
    count_for_no_ents = 0
    count = 0
    synerr = 0
    for item in tqdm(dataset):
        iid = item["pmid"]
        if item["passages"][0]["type"] == "title":
            title = item["passages"][0]["text"][0]
        abstract = item["passages"][1]["text"][0]
        text = title + abstract
        sent_text = []
        
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)

        sent_response = {}
        sent_messages = {}
        extracted_abs = extracted_entities[iid]

        for sent_id, text in enumerate(sent_text):

            messages = []
            # remove human here? 
            messages += [f"<human>: {prompt}"]
            messages += [f"<human>: {text}" ]

            try:
                extracted_sent = extracted_abs[str(sent_id)]
                if isinstance(extracted_sent, dict):
                    messages += [f"<bot>:",json.dumps(
                                {
                                    k: v
                                    for k, v in extracted_sent.items()
                                    if k in ("entity")
                                }
                            ),
                    ]
                    list_to_retrieve = []
                    for types, entities in extracted_sent.items():
                        if isinstance(entities, list):
                            for each_entity in entities:
                                list_to_retrieve += each_entity
                        list_to_retrieve += entities

                else:
                    messages += [f"<bot>: {'entity'}: []"]
                    synerr += 1
                    list_to_retrieve = [] 
                    content = ""
   
            except Exception as e:
                messages += [f"<bot>: {'entity'}: []"]
                synerr += 1
                list_to_retrieve = []
                count_for_no_ents += 1 


            biomedical_entities = linker.extract_noun_phrases_remove_repeats(text, list_to_retrieve)
        
            ent_definitions = retriever.link_with_source(list_to_retrieve)
            noun_definitions = retriever.link_with_source(biomedical_entities)

            content = ""
            for key, value in ent_definitions.items():
                content += f"{key}:{value}\n"

            content_noun = ""
            for key, value in noun_definitions.items():
                content_noun += f"{key}:{value}\n"

            if content == "":
                if content_noun == "":
                    messages += [f"<human>: {common.PROMPT_END_NO_DEF}"]
                else: 
                    messages += [f"<human>: {common.PROMPT_ENT + content_noun +  common.PROMPT_END}"]
            else:
                if content_noun == "":
                    messages += [f"<human>: {common.PROMPT_ENT + content +  common.PROMPT_END}"]
                else:   
                    messages += [f"<human>: {common.PROMPT_ENT + content  + content_noun +  common.PROMPT_END + text}"]


            response = llama_call.generate_text(messages,  temperature=0, max_tokens=256)
            

            sent_response[sent_id] = response 
            sent_messages[sent_id] = messages

            
        all_responses[iid]["responses"] = sent_response
        all_responses[iid]["messages"] = sent_messages
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs_abl/medm/llama/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_ret/medm/llama/zs_wiki_def_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)

    print(count_for_no_ents)


if __name__ == "__main__":
    main()
