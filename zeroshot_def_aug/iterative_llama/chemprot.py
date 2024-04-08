import json
import os
from tqdm import tqdm
from ipdb import set_trace
from calls import retrieval
from collections import defaultdict
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import utils
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from constants import OUTPUT_DIR

HUMAN_PROMPT = "<human>:"
AI_PROMPT = "<bot>:"

ENTITY_TYPES = ["chemical", "protein"]

def main():
    all_responses = defaultdict(dict)
    dataset = load_dataset("bigbio/chemprot")["test"]
    output = OUTPUT_DIR / "final_zs_subsample/chemprot/llama/zs_text_json_100_for_retrieval.json"

    extracted_entities = json.load(open(output))
    retriever = retrieval.KnowledgeRetrieval()

    count = 0
    synerr = 0
    keyerr = 0
    for item in tqdm(dataset):
        iid = item["pmid"]
        text = item["text"]
        sent_text = []
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            sentence = text[start:end]
            sent_text.append(sentence)

        extracted_abs = extracted_entities[iid]
        sent_response = {}
        sent_messages = {}
        for sent_id, text in enumerate(sent_text):
 
            try:
                extracted_sent = extracted_abs[str(sent_id)]
                #extracted_sent = ast.literal_eval(extracted_sent_raw.strip())
            except SyntaxError:
                synerr += 1
                extracted_sent = {"chemical": [], "protein": []}

            except KeyError:
                keyerr += 1
                extracted_sent = {"chemical": [], "protein": []}

            # assign extracted_sent t0 a new variable ;
            # use this to update the entities and save as the final output

            list_to_retrieve = []
            for _, entities in extracted_sent.items():
                list_to_retrieve += entities

            ent_definitions = dict(retriever.link_with_umls(list_to_retrieve))
            noun_definitions = dict(retriever.extract_noun_phrases_and_link_with_umls_remove_repeats(text, list_to_retrieve))
            # Here I will have all the entities that are extracted and also noun phrases with definitions,  I need to iterate over these
            final_output = defaultdict(list)
            for entity_type, entity_list in extracted_sent.items():
                final_entity_list = []

                for entity in entity_list:
                    if not ent_definitions.get(entity):
                        final_entity_list.append(entity)
                        continue

                    definition = ent_definitions[entity][1]
                    if utils.is_entity_in_entity_type(entity=entity, definition=definition, entity_type=entity_type):
                        # remove the entity from extracted sent
                        final_entity_list.append(entity)

                final_output[entity_type] = final_entity_list
                
            for noun_phrase, definition in noun_definitions.items():
                query_message = [f"{HUMAN_PROMPT}{utils.BOOLEAN_NOUN_PROMPT.format(noun_phrase=noun_phrase, definition=definition, entity_types=ENTITY_TYPES)}"]
                # query_message = [f"{HUMAN_PROMPT}{utils.BOOLEAN_NOUN_PROMPT_NO_DEF.format(noun_phrase=noun_phrase, entity_types=ENTITY_TYPES)}"]
                
                bool_response = utils.get_bool_openai_response(query_message)
                if bool_response:
                    noun_phrase_types = utils.get_entity_types_for_entity(noun_phrase, definition, ENTITY_TYPES)
      
                    if noun_phrase_types:
                        for entity_type in noun_phrase_types:
                            final_output[entity_type].append(noun_phrase)

            sent_response[sent_id] = dict(final_output)
            sent_messages[sent_id] = extracted_sent

        all_responses[iid]["responses"] = sent_response
        all_responses[iid]["messages"] = sent_messages
        
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            path = OUTPUT_DIR / "/final_zs_ret/chemprot/llama/"
            isExist = os.path.exists(path)
            if not isExist:
                print("created path")
                os.makedirs(path)
            with open(
                OUTPUT_DIR / f"/final_zs_ret/chemprot/llama/zs_ques_ret_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
