import re
import io
import ast
import json
import random
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from typing import List, Dict, Any
from collections import defaultdict
from nltk.tokenize.punkt import PunktSentenceTokenizer

ENTITY_TYPES_MAP = {
    "Participant": "population",
    "Intervention": "intervention",
    "Comparator": "comparator",
    "Outcome": "outcome",
}

class CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        # Here you can serialize your object depending of its type
        # or you can define a method in your class which serializes the object
        if isinstance(o, set):
            return list(o)  # Or another method to serialize it
        else:
            return json.JSONEncoder.encode(self, o)


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0.0
    all_syn_err = 0
    all_key_err = 0
    all_type_err = 0
    all_val_err = 0
    all_attr_err = 0
    all_idx_err = 0
    num_extractions = 0
    predicted_entities = {}
    all_errors = []
    for doc_id in extractions:
        # for each doc
        extracted_entities_raw = extractions[doc_id]
        #extracted_messages = extractions[doc_id]["messages"]
        gold_entities = entities_by_pmid[doc_id]
        gold_text = text_by_pmid[doc_id]

        sentence_gold_entities = sentence_wise(gold_text, gold_entities)

        # For each document, postprocess extracted entities
        (
            extracted_entities,
            correct_preds,
            doc_num_extractions,
            syn_err,
            key_err,
            val_err,
            type_err,
            attr_err,
            idx_err,
            error_sents,
        ) = postprocess_entity_outputs(extracted_entities_raw, sentence_gold_entities)

        predicted_entities[doc_id] = extracted_entities
        # For every entity type, send it to strict_match with all entities of same type in gold
        overall_strict_correct += correct_preds
        all_syn_err += syn_err
        all_key_err += key_err
        all_val_err += val_err
        all_type_err += type_err
        all_attr_err += attr_err
        all_idx_err += idx_err
        num_extractions += doc_num_extractions

    
    overall_strict_prec = overall_strict_correct / num_extractions

    total_gold_extractions = 0
    for idx in entities_by_pmid:
        length = len(entities_by_pmid[idx])
        total_gold_extractions += length

    overall_strict_rec = overall_strict_correct / total_gold_extraction
    overall_strict_prec = round(overall_strict_prec, 4)
    overall_strict_rec = round(overall_strict_rec, 4)

    print(overall_strict_prec)
    print(overall_strict_rec)
    print(all_syn_err)
    print(all_key_err)
    print(all_type_err)
    print(all_val_err)
    print(all_attr_err)
    print(all_idx_err)
    
    return overall_strict_prec, overall_strict_rec


def _is_entity_in_sentence(data_dict, offset):
    if data_dict["start"] >= offset[0] and data_dict["end"] <= offset[1]:
        return True
    else:
        return False


def sentence_wise(gold_text, gold_entities):
    """
    This function works with gold extrations and looks at the sentence offsets saved in gold_text and buckets the entities from gold_entities into sentences.
    """
    abs_gold_entities = {}
    offset_list = gold_text["sent_offsets"]
    idx = 0

    sorted_list = sorted(gold_entities, key=lambda x: x["start"])
    for i, sent_offset in enumerate(offset_list):
        sent_gold_entities = []
        while idx < len(sorted_list) and _is_entity_in_sentence(
            sorted_list[idx], sent_offset
        ):
            sent_gold_entities.append(sorted_list[idx])
            idx += 1

        abs_gold_entities[str(i)] = sent_gold_entities
    return abs_gold_entities


def postprocess_entity_outputs(extracted_entities, gold_extraction):
    """
    This function postprocesses the predicted extractions, gets it back into the json format and also takes the gold extractions and maps them with the predicted.
    The counter keeps the track of number of correct exstrations for calculating precision and recall.
    """
    postprocessed_entities = {}
    correct_preds = 0
    total_num_extraction = 0
    syn_err = 0
    key_err = 0
    val_err = 0
    type_err = 0
    attr_err = 0
    idx_err = 0
    error_sents = set()
    for sentence_number, extraction in extracted_entities.items():
    
        
        extraction = extraction.replace("\n", "")
        " ".join(extraction.split())
        extraction = extraction.strip()

        if extraction[0] == "[":
            extraction = extraction[1:]

        if extraction[-1] == "]":
            extraction = extraction[:-1]

        
        print("--------")
        print(extraction)

        if "\n<human>" in extraction:
            extraction = extraction.split("\n<human>")[0]

        if "<human>" in extraction:
            extraction = extraction.split("<human>")[0]

        if "</human:>" in extraction:
            extraction = extraction.split("</human:>")[0]

        if "</human>" in extraction:
            extraction = extraction.split("</human>")[0]
        
        if "<\n/human>" in extraction:
            extraction = extraction.split("<\n/human>")[0]

        if "Sentence" in extraction:
            extraction = extraction.split("Sentence")[0]

        if "\n['" in extraction:
            extraction = extraction.split("\n['")[0]
        
        if '\n["' in extraction:
            extraction = extraction.split('\n["')[0]
        
        if "\n```" in extraction:
            extraction = extraction.split("\n```")[0]

        extraction = extraction.strip()

        if "]}\n" in extraction:
            extraction = extraction.split("]}\n")[0] + "]}"
        if "Here " in extraction or "here" in extraction:
            try:
                if "format:" in extraction:
                    extraction = extraction.split("format:")[1]
                elif "extracted:" in extraction:
                    extraction = extraction.split("extracted:")[1]
                elif "extracted entities:" in extraction:
                    extraction = extraction.split("extracted entities:")[1]
                elif "sentence:" in extraction:
                    extraction = extraction.split("sentence:")[1]
                elif "UMLS:" in extraction:
                    extraction = extraction.split("UMLS:")[1]
                elif "Metathesaurus:" in extraction:
                    extraction = extraction.split("Metathesaurus:")[1]

            except IndexError as e:
                True # 

        if "```json" in extraction:
            extraction = extraction.split("```json")[1].split("```")[0]

        if "```python" in extraction:
            extraction = extraction.split("```python")[1].split("```")[0]
        if "base:" in extraction:
            extraction = extraction.split("base:")[1]
        if "are:" in extraction:
            extraction = extraction.split("are:")[1]
        if "is:" in extraction:
            extraction = extraction.split("is:")[1]
        if "would be:" in extraction:
            extraction = extraction.split("would be:")[1]
        if "abstract:" in extraction:
            extraction = extraction.split("abstract:")[1]
        if "information:" in extraction:
            extraction = extraction.split("information:")[1]
        if "json:" in extraction:
            extraction = extraction.split("json:")[1]
        if "JSON:" in extraction:
            extraction = extraction.split("JSON:")[1]
        if "object:" in extraction:
            extraction = extraction.split("object:")[1]
        if "elements:" in extraction:
            extraction = extraction.split("elements:")[1]
        if "types:" in extraction:
            extraction = extraction.split("types:")[1]
        if "output:" in extraction:
            extraction = extraction.split("output:")[1]
        if "entities:" in extraction:
            extraction = extraction.split("entities:")[1]
        if "mentions:" in extraction:
            extraction = extraction.split("mentions:")[1]
        if "proteins:" in extraction:
            extraction = extraction.split("proteins:")[1]
   
        processed_extractions = {
            "population": [],
            "intervention": [],
            "comparator": [],
            "outcome": [],
        }
        
        
        if extraction[0] == "(":
            extraction = extraction[1:]
        if extraction[-1] == ")":
            extraction = extraction[:-1]
        
        try:
            extraction = ast.literal_eval(extraction)
            if isinstance(extraction, tuple):
                extraction = extraction
            if not isinstance(extraction, dict):
                # print(extraction)
                continue
        
            processed_extractions = {extraction['type'].lower(): [extraction['text']]}

            # for claude the eextractions in values are not in a list but
            # just strings, please make v enclised in a list
            """
            for k, v in extraction.items():
                for pico in processed_extractions.keys():
                    
                    if k == "comparator" and pico == "comparator":
                        #for def change it to [v]
                        # if isinstance(v, list):
                        processed_extractions["intervention"] += v
                        # else:
                        #     processed_extractions[pico] += [v]
                    elif pico in k.lower():
                        #if isinstance(v, list):
                        processed_extractions[pico] += v
                        # else:
                        #     processed_extractions[pico] += [v]
            """

            num_extractions = sum([len(v) for k, v in processed_extractions.items()])
            total_num_extraction += num_extractions
            compared_indices = set()
            
            

            for i in range(len(gold_extraction[sentence_number])):
                text = gold_extraction[sentence_number][i]["text"]
                gold_type = gold_extraction[sentence_number][i]["annotation_type"]

                data_type = ENTITY_TYPES_MAP.get(gold_type)

                if not data_type:
                    print(gold_type)
                    

                found = False
                try:
                    for j in range(len([processed_extractions[data_type]])):
                        if (i, j) not in compared_indices:
                            # for claude text json, needs [0] after [j] in the next line
                            if processed_extractions[data_type]:
                                if processed_extractions[data_type][j] == text:
                                    correct_preds += 1
                                    found = True
                                    compared_indices.add((i, j))
                                    compared_indices.add((j, i))

                except IndexError as e:
                    # print(e)
                    idx_err += 1
                    True # 
                pass

                if not found:
                    error_sents.add(sentence_number)

                print("********************")
                print(processed_extractions)
            postprocessed_entities[sentence_number] = processed_extractions

        except SyntaxError as e:
            # This code will be executed if a SyntaxError exception is raised
            # print(e)
            # print(extraction)
            #print("This is a Syntax error")
            syn_err += 1
            pass

        except ValueError as e:
            val_err += 1
            pass

    return (
        postprocessed_entities,
        correct_preds,
        total_num_extraction,
        syn_err,
        key_err,
        val_err,
        type_err,
        attr_err,
        idx_err,
        error_sents,
    )



def create_subsample_dict(dict1, dict2):
    dict3 = {}
    for each_passage in dict1:
        doc_id = each_passage["doc_id"]
        if doc_id in dict2:
            dict3[doc_id] = dict2[doc_id]


    return dict3


def main(output_file):
    print("Loading dataset...")
    
    dataset = load_dataset("bigbio/ebm_pico", split="test")
    extracted_entities = json.load(open(output_file))

    entities_by_pmid = {}
    text_by_pmid = {}

    for idx, example in enumerate(dataset):
        entities = []
        pmid = example["doc_id"]
        # text offset
        content = example["text"]
        sent_offsets = []
        # Adding in abstract dict, doesn't include title
        for start, end in PunktSentenceTokenizer().span_tokenize(content):
            cur_offset = [start, end]
            sent_offsets.append(cur_offset)

        text_by_pmid[pmid] = {}
        text_by_pmid[pmid]["text"] = content
        text_by_pmid[pmid]["sent_offsets"] = sent_offsets

        all_list = []
        all_dict_list = []
        for each in example["entities"]:
            each_dict = {}
            start = each["start"]
            end = each["end"]
            text = each["text"]
            atype = each["annotation_type"]
            if (start, end, atype) not in all_list:
                all_list.append((start, end, atype))

                each_dict["text"] = text
                each_dict["annotation_type"] = atype
                each_dict["start"] = start
                each_dict["end"] = end
                all_dict_list.append(each_dict)

        entities_by_pmid[pmid] = all_dict_list
    

    prec, rec = eval_entity_extraction(extracted_entities, text_by_pmid, entities_by_pmid)

    return prec, rec


if __name__ == "__main__":
    main()
