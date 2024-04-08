import openai
import os
import json
import wiki
from calls import openai_call
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


def main():
    all_responses = defaultdict(dict)
    prompt = ncbi.GPT_TEXT
    dataset = load_dataset("bigbio/ncbi_disease", split="test")
    output = OUTPUT_DIR / "final_zs_subsample/ncbi/gpt4/zs_text_json_100_for_retrieval.json"

    extracted_entities = json.load(open(output))
    linker = retrieval.KnowledgeRetrieval()
    retriever = wiki.KnowledgeRetrieval()

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

        for sent_id, text in enumerate(sent_text):
            try:
                extracted_sent = extracted_abs[str(sent_id)]
                #extracted_sent = ast.literal_eval(extracted_sent_raw.strip())
            except SyntaxError as e:
                synerr += 1
                
                pass

            except KeyError as e:
                keyerr += 1
                

            messages = []
            messages += [{"role": "user", "content": prompt}]
            
            messages += [
                {"role": "user", "content": f"Sentence: {text}"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            k: v
                            for k, v in extracted_sent.items()
                            if k in ("diseases")
                        }
                    ),
                },
            ]
            
            list_to_retrieve = []
            for types, entities in extracted_sent.items():
                list_to_retrieve += entities

            biomedical_entities = linker.extract_noun_phrases_remove_repeats(text, list_to_retrieve)
        
            ent_definitions = retriever.link_with_source(list_to_retrieve)
            noun_definitions = retriever.link_with_source(biomedical_entities)
            # call to retrieve defintions 

            content = ""
            for key, value in ent_definitions.items():
                content += f"{key}:{value}\n"

            content_noun = ""
            for key, value in noun_definitions.items():
                content_noun += f"{key}:{value}\n"

            if content == "":
                if content_noun == "":
                    messages += [{"role": "user", "content": common.PROMPT_END_NO_DEF + "\nOutput:" }]
                else: 
                    messages += [{"role": "user", "content": common.PROMPT_NOUN + content_noun +  common.PROMPT_END+ "\n Sentence: " + text + "\nOutput:" }]
            else:
                if content_noun == "":
                    messages += [{"role": "user", "content": common.PROMPT_RET + content +  common.PROMPT_END+ "\n Sentence: " + text + "\nOutput:" }]
                else:   
                    messages += [{"role": "user", "content": common.PROMPT_RET + content + common.PROMPT_NOUN + content_noun +  common.PROMPT_END+ "\n Sentence: " + text + "\nOutput:" }]
     
            # increase the num of tokens
            response = openai_call.generate_text(
                messages, schema, temperature=0, max_tokens=256
            )
            
            sent_response[sent_id] = response   
            sent_messages[sent_id] = messages


        all_responses[iid]["responses"] = sent_response
        all_responses[iid]["messages"] = sent_messages
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs_abl/ncbi/gpt4/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_zs_abl/ncbi/gpt4/zs_wiki_def_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
