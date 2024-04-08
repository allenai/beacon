import openai
import os
import json
import random
from calls import openai_call
import utils
from calls import retrieval
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset
from collections import defaultdict
from nltk.tokenize.punkt import PunktSentenceTokenizer
from prompts import common, ncbi
from constants import OUTPUT_DIR

openai.api_key = os.getenv("OPENAI_API_KEY")

schema = {
    "type": "object",
    "properties": {
        "diseases": {
            "type": "array",
            "description": "list of all diseases",
            "items": {"type": "string"},
        },
    },
    "required": ["diseases"],
}


def run_fewshot(k, seed): 
    print(seed, k)
    all_responses = defaultdict(dict)
    prompt = ncbi.GPT_TEXT
    # ncbi full dataset is only 100
    dataset = load_dataset("bigbio/ncbi_disease", split="test")
    output_formatter = utils.OutputFormatter(dataset="ncbi", seed=seed)
    output = OUTPUT_DIR / f"/final_fs_subsample/ncbi/gpt/zs_text_json_5shot_seed{seed}_100.json"

    extracted_entities = json.load(open(output))
    retriever = retrieval.KnowledgeRetrieval()

    count = 0
    synerr = 0
    keyerr = 0
    for item in tqdm(dataset):
        iid = item["pmid"]
        title = item["title"]
        abstract = item["abstract"]
        # space even in zeroshot for retrieval
        text = title + " " + abstract
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)
        sent_response = {}
        sent_messages = {}
        extracted_abs = extracted_entities[iid]

        for sent_id, this_text in enumerate(sent_text):
            
            try:
                extracted_sent = extracted_abs[str(sent_id)]
                #extracted_sent = ast.literal_eval(extracted_sent_raw.strip())
            except SyntaxError as e:
                synerr += 1
                extracted_sent = {"diseases" : []}

            except KeyError as e:
                keyerr += 1
                extracted_sent = {"diseases" : []}
                
            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            messages += [{"role": "user", "content": prompt}]
            
            for i in range(len(formatted_shots)):

                shot_text = formatted_shots[i]["text"]
                messages += [{"role": "user", "content": shot_text}]
                list_to_retrieve = []
                list_to_retrieve += formatted_shots[i]["diseases"] 
                
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
                        messages += [{"role": "user", "content": common.PROMPT_END_NO_DEF + "\nOutput:" }]
                    else: 
                        messages += [{"role": "user", "content": common.PROMPT_END + content_noun + "\nOutput:" }]
                else:
                    if content_noun == "":
                        messages += [{"role": "user", "content": common.PROMPT_ENT + content + "\nOutput:" }]
                    else:   
                        messages += [{"role": "user", "content": common.PROMPT_ENT + content + content_noun + "\nOutput:" }]

                # Here are the definitions
                #for every shot we dont have to give it json
                # in every formatted shot, we need to look up and append definitions

                messages += [
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                k: v
                                for k, v in formatted_shots[i].items()
                                if k in ("diseases")
                            }
                        ),
                    },
                ]

            messages += [{"role": "user", "content": this_text}]            
            list_to_retrieve_text = []
            for types, entities in extracted_sent.items():
                list_to_retrieve_text += entities
            
            noun_definitions = retriever.extract_noun_phrases_and_link_with_umls_remove_repeats(this_text, list_to_retrieve)
            ent_definitions = retriever.link_with_umls(list_to_retrieve)

            content = ""
            for key, value in ent_definitions.items():
                content += f"{key}: {value}\n"

            content_noun = ""
            for key, value in noun_definitions.items():
                content_noun += f"{key}: {value}\n"

            
            if content == "":
                if content_noun == "":
                    messages += [{"role": "user", "content": common.PROMPT_END_NO_DEF + "\nOutput:" }]
                else: 
                    messages += [{"role": "user", "content": common.PROMPT_ENT + content_noun + common.PROMPT_END + "\n Sentence: " + this_text + "\nOutput:" }]
            else:
                if content_noun == "":
                    messages += [{"role": "user", "content": common.PROMPT_ENT + content + common.PROMPT_END + "\n Sentence: " + this_text + "\nOutput:" }]
                else:   
                    messages += [{"role": "user", "content": common.PROMPT_ENT + content + content_noun + common.PROMPT_END + "\n Sentence: " + this_text + "\nOutput:" }]
                    
            response = openai_call.generate_text(
                messages, schema, temperature=0, max_tokens=2048
            )

            sent_response[sent_id] = response   
            #sent_messages[sent_id] = messages
    
        all_responses[iid]["responses"] = sent_response
        #all_responses[iid]["messages"] = sent_messages
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_fs_ret/ncbi/gpt/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs_ret/ncbi/gpt/fs_stC_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


def main():
    pass

if __name__ == "__main__":
    main()
