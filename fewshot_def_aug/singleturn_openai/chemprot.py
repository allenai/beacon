import json
import random
import utils
import os
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from calls import openai_call
from calls import retrieval
from prompts import common, chemprot
from constants import OUTPUT_DIR

schema = {
    "type": "object",
    "properties": {
        "chemicals": {
            "type": "array",
            "description": "List of all the chemicals.",
            "items": {"type": "string"},
        },
        "proteins": {
            "type": "array",
            "description": "List of all the diseases.",
            "items": {"type": "string"},
        },
    },
    "required": ["chemicals", "proteins"],
}


def run_fewshot(k, seed):
    print(seed, k)
    all_responses = {}
    prompt = chemprot.GPT_TEXT
    dataset = load_dataset("bigbio/chemprot")["test"]
    
    output = OUTPUT_DIR / f"/final_fs_subsample/chemprot/gpt/zs_text_json_5shot_seed{seed}_100.json"
    extracted_entities = json.load(open(output))

    count = 0
    keyerr = 0 
    synerr = 0
    output_formatter = utils.OutputFormatter(dataset="chemprot", seed=seed)
    retriever = retrieval.KnowledgeRetrieval()

    for item in tqdm(dataset):
        iid = item["pmid"]
        text = item["text"]
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
                extracted_sent = {"chemical" : [], "protein" : []}
    

            except KeyError as e:
                keyerr += 1
                extracted_sent = {"chemical" : [], "protein" : []}

            key_id = iid + "-" + str(sent_id)
            formatted_shots = output_formatter.format_output(key_id=key_id, k=k)
            random.shuffle(formatted_shots)
            messages = []
            messages += [{"role": "user", "content": prompt}]
            for i in range(len(formatted_shots)):
                
                shot_text = formatted_shots[i]["text"]
                messages += [{"role": "user", "content": shot_text}]
                list_to_retrieve = []
                list_to_retrieve += formatted_shots[i]["chemicals"] 
                list_to_retrieve += formatted_shots[i]["proteins"]

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
                                if k in ("chemicals", "proteins")
                            }
                        ),
                    },
                ]
                
            # here I use the extraced entities defintions

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
                messages, schema, temperature=0, max_tokens=128
            )
            sent_response[sent_id] = response
            
        all_responses[iid] = sent_response
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_fs_ret/chemprot/gpt/"
            isExist = os.path.exists(path)
            if not isExist:
                print("creating path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_fs_ret/chemprot/gpt/fs_stC_{k}shot_seed{seed}_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)
    
    print(synerr)
    print(keyerr)

def main():
    pass


if __name__ == "__main__":
    main()
