import os
import json
from tqdm import tqdm
from ipdb import set_trace
from collections import defaultdict
from datasets import load_dataset, load_from_disk
from calls import claude_call
from nltk.tokenize.punkt import PunktSentenceTokenizer
from calls import retrieval
from anthropic import HUMAN_PROMPT, AI_PROMPT
from prompts import common, cdr
from constants import OUTPUT_DIR

def main():
    all_responses = defaultdict(dict)
    prompt = cdr.GPT_TEXT
    dataset = load_dataset("bigbio/bc5cdr")["test"]
    output = OUTPUT_DIR / "final_zs/cdr/claude/zs_text_json_100_for_retrieval.json"

    extracted_entities = json.load(open(output))
    retriever = retrieval.KnowledgeRetrieval()
    # after reading these and matching
    count = 0
    synerr = 0
    keyerr = 0
    for item in tqdm(dataset):
        iid = item["passages"][0]["document_id"]
        if item["passages"][0]["type"] == "title":
            title = item["passages"][0]["text"]
        abstract = item["passages"][1]["text"]
        # no space for zeroshot
        text = title +  " " + abstract
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)

        # look up sentence wise extractions in
        extracted_abs = extracted_entities[iid]
        sent_response = {}
        sent_messages = {}
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
            messages += [f"{HUMAN_PROMPT}{prompt}"]
            messages += [
                f"{HUMAN_PROMPT}{text}" 
                f"{AI_PROMPT}",json.dumps(
                        {
                            k: v
                            for k, v in extracted_sent.items()
                            if k in ("chemical", "disease")
                        }
                    ),
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
                    messages += [f"{HUMAN_PROMPT}{common.PROMPT_END_NO_DEF}"]
                else: 
                    messages += [f"{HUMAN_PROMPT}{common.PROMPT_NOUN + content_noun +  common.PROMPT_END + text}"]
            else:
                if content_noun == "":
                    messages += [f"{HUMAN_PROMPT}{common.PROMPT_RET + content +  common.PROMPT_END + text}"]
                else:   
                    messages += [f"{HUMAN_PROMPT}{common.PROMPT_RET + content + common.PROMPT_NOUN + content_noun +  common.PROMPT_END + text}"]


            response = claude_call.generate(messages)
            
            sent_response[sent_id] = response   
            sent_messages[sent_id] = messages

        all_responses[iid]["responses"] = sent_response
        all_responses[iid]["messages"] = sent_messages
        # time.sleep(3)
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_ret/cdr/claude/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_ret/cdr/claude/zs_stC_ret_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)

            print("Saved")

    print(synerr)
    print(keyerr)


if __name__ == "__main__":
    main()
