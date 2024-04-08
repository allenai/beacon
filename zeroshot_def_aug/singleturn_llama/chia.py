import json
from calls import llama_call
from tqdm import tqdm
import os
from ipdb import set_trace
from datasets import load_from_disk
from collections import defaultdict
from calls import retrieval
import wiki
from prompts import common, chia
from constants import OUTPUT_DIR

def main():
    all_responses = defaultdict(dict)
    prompt = chia.TEXT_JSON
    dataset = load_from_disk(
        ""replace with the chia data split path"/val.json"
    )
    # rerun zeroshot og for retrieval 
    output = OUTPUT_DIR / "final_zs_subsample/chia/llama/zs_text_json_100_for_retrieval.json"
    
    extracted_entities = json.load(open(output))

    #extracted_entities = create_subsample_dict(dataset, all_extracted_entities)
    print(len(extracted_entities))
    
    linker = retrieval.KnowledgeRetrieval()
    retriever = wiki.KnowledgeRetrieval()

    count = 0
    count_for_no_ents = 0 
    synerr = 0
    keyerr = 0
    idxerr = 0 
    for item in tqdm(dataset):
        iid = item["id"]
        all_text = item["text"]
        sent_text = filter(None, all_text.split("\n"))
        # sent_text = nltk.sent_tokenize(all_text)
        extracted_abs = extracted_entities[iid]
        sent_response = {}
        sent_messages = {}
        # change criterion to calibration_prompt
        if item["id"][-3:] == "exc":
            criterion = "Given the Exclution criterion for a clinical trial. \n"
        else:
            criterion = "Given the Inclusion criterion for a clinical trial. \n"
        for sent_id, text in enumerate(sent_text):
            messages = []
            # remove human here? 
            messages += [f"<human>: {prompt}"]
            messages += [f"<human>: {text}" ]
            
            try:
                extracted_sent = extracted_abs[str(sent_id)]

            except Exception as e:
                extracted_sent = {'condition': [],
                'device': [],
                'drug': [],
                'measurement' : [],
                'observation' : [],
                'person' : [],
                "procedure" : [],
                "visit": [],
                "temporal" : [],
                "value" : [],
                "scope" : [],
                "negation" : [],
                "qualifier" : [],
                "multiplier" : [],
                "reference_point" : [],
                "mood": []}
                list_to_retrieve = []
                count_for_no_ents += 1


            messages += [f"<bot>:",json.dumps(
                            {
                                k: v
                                for k, v in extracted_sent.items()
                                if k in ("value", "reference_point","device",
                                "multiplier", "condition", "temporal",
                                "person", "drug", "negation", "measurement",
                                "procedure", "visit", "mood", "qualifier",
                                "observation", "scope")
                            }
                        ),
                ]
            
            list_to_retrieve = []
            for types, entities in extracted_sent.items():
                list_to_retrieve += entities

            biomedical_entities = linker.extract_noun_phrases_remove_repeats(text, list_to_retrieve)
        
            ent_definitions = retriever.link_with_source(list_to_retrieve)
            noun_definitions = retriever.link_with_source(biomedical_entities)

            content = ""
            for key, value in ent_definitions.items():
                content += f"{key}:\n{value}\n"

            content_noun = ""
            for key, value in noun_definitions.items():
                content_noun += f"{key}:\n{value}\n"

            if content == "":
                if content_noun == "":
                    messages += [f"<human>: {common.PROMPT_END_NO_DEF}"]
                else: 
                    messages += [f"<human>: {common.PROMPT_ENT+ content_noun +  common.PROMPT_END}"]
            else:
                if content_noun == "":
                    messages += [f"<human>: {common.PROMPT_ENT+ content +  common.PROMPT_END}"]
                else:   
                    messages += [f"<human>: {common.PROMPT_ENT+ content  + content_noun +  common.PROMPT_END+ text}"]


            response = llama_call.generate_text(messages,  temperature=0, max_tokens=256)
            
            
            sent_response[sent_id] = response  
            sent_messages[sent_id] = messages 


        all_responses[iid]["responses"] = sent_response
        all_responses[iid]["messages"] = sent_messages

        count += 1
        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs_abl/chia/llama/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_ret/chia/llama/zs_wiki_def_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)

        print(count_for_no_ents)

if __name__ == "__main__":
    main()
