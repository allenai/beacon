import json
from calls import openai_call
from tqdm import tqdm
import os
from ipdb import set_trace
from datasets import load_from_disk
from collections import defaultdict
from calls import retrieval
from prompts import common, chia
from constants import OUTPUT_DIR

schema = {
    "type": "object",
    "properties": {
        "Value": {
            "type": "array",
            "description": "List of all the values.",
            "items": {"type": "string"},
        },
        "Reference_points": {
            "type": "array",
            "description": "List of all the reference points.",
            "items": {"type": "string"},
        },
        "Devices": {
            "type": "array",
            "description": "List of all the devices.",
            "items": {"type": "string"},
        },
        "Multiplier": {
            "type": "array",
            "description": "List of all the multipliers.",
            "items": {"type": "string"},
        },
        "Condition": {
            "type": "array",
            "description": "List of all the conditions.",
            "items": {"type": "string"},
        },
        "Temporal": {
            "type": "array",
            "description": "List of all the temporal info.",
            "items": {"type": "string"},
        },
        "Person": {
            "type": "array",
            "description": "List of all the Person.",
            "items": {"type": "string"},
        },
        "Drug": {
            "type": "array",
            "description": "List of all the Drugs.",
            "items": {"type": "string"},
        },
        "Negation": {
            "type": "array",
            "description": "List of all the Negations.",
            "items": {"type": "string"},
        },
        "Measurement": {
            "type": "array",
            "description": "List of all the Measurements.",
            "items": {"type": "string"},
        },
        "Procedure": {
            "type": "array",
            "description": "List of all the Procedures.",
            "items": {"type": "string"},
        },
        "Visit": {
            "type": "array",
            "description": "List of all the Visits.",
            "items": {"type": "string"},
        },
        "Mood": {
            "type": "array",
            "description": "List of all the Moods.",
            "items": {"type": "string"},
        },
        "Qualifier": {
            "type": "array",
            "description": "List of all the Qualifiers.",
            "items": {"type": "string"},
        },
        "Observation": {
            "type": "array",
            "description": "List of all the Observations.",
            "items": {"type": "string"},
        },
        "Scope": {
            "type": "array",
            "description": "List of all the Scopes.",
            "items": {"type": "string"},
        },
    },
    "required": [
        "Value",
        "Reference_point",
        "Device",
        "Multiplier",
        "Condition",
        "Temporal",
        "Person",
        "Drug",
        "Negation",
        "Measurement",
        "Procedure",
        "Visit",
        "Mood",
        "Qualifier",
        "Observation",
        "Scope",
    ],
}


def main():
    all_responses = defaultdict(dict)
    prompt = chia.GPT_TEXT
    dataset = load_from_disk(
        ""replace with the chia data split path"/val.json"
    )
    output = OUTPUT_DIR / "final_zs/chia/gpt4/zs_text_json_100_for_retrieval.json"
    
    extracted_entities = json.load(open(output))
    retriever = retrieval.KnowledgeRetrieval()

    count = 0
    synerr = 0
    keyerr = 0

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
            try:
                extracted_sent = extracted_abs[str(sent_id)]
                #extracted_sent = ast.literal_eval(extracted_sent_raw)
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
                            if k in ( "value", "reference_point","device",
                                    "multiplier", "condition", "temporal",
                                    "person", "drug", "negation", "measurement",
                                    "procedure", "visit", "mood", "qualifier",
                                    "observation", "scope",)
                        }
                    ),
                },
            ]
            
            list_to_retrieve = []
            for types, entities in extracted_sent.items():
                list_to_retrieve += entities


            noun_definitions = retriever.extract_noun_phrases_and_link_with_umls_remove_repeats(text, list_to_retrieve)
            ent_definitions = retriever.link_with_umls(list_to_retrieve)
            # call to retrieve defintions 

            content = ""
            for key, value in ent_definitions.items():
                content += f"{key}:\n{value}\n"

            content_noun = ""
            for key, value in noun_definitions.items():
                content_noun += f"{key}:\n{value}\n"

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
        
            response = openai_call.generate_text(
                messages, schema, temperature=0, max_tokens=512
            )
            
            sent_response[sent_id] = response  
            sent_messages[sent_id] = messages 

        all_responses[iid]["responses"] = sent_response
        all_responses[iid]["messages"] = sent_messages

        count += 1
        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_ret/chia/gpt4/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_ret/chia/gpt4/zs_stC_ret_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
