import re
import io
import ast
import json
import nltk
import random
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from typing import List, Dict, Any
from collections import defaultdict
from nltk.tokenize.punkt import PunktSentenceTokenizer

all_syntax_err = 0

ENTITY_TYPES = [
    "Event",
    "Activity",
    "Behavior",
    "Social Behavior",
    "Individual Behavior",
    "Daily  or Recreational Activity",
    "Occupational Activity",
    "Health care Activity",
    "Research Activity",
    "Government or Regulatory Activity",
    "Educational Activity",
    "Machine Activity",
    "Phenomenon or Process",
    "Injury or Poisoning",
    "Human-caused Phenomenon or Process",
    "Environmental Effects of Humans",
    "Natural Phenomenon or Process",
    "Biological Function",
    "Entity",
    "Physical Object",
    "Organism",
    "Virus",
    "Bacterium",
    "Archaeon",
    "Eukaryote",
    "Anatomical Structure",
    "Manufactured Object",
    "Medical Device",
    "Research Device",
    "Clinical Drug",
    "Substance",
    "Body Structure",
    "Chemical",
    "Food",
    "Conceptual Identity",
    "Organism Attribute",
    "Clinical Attribute",
    "Finding",
    "Idea or Concept",
    "Temporal Concept",
    "Qualitative Concept",
    "Quantitative Concept",
    "Spatial Concept",
    "Functional Concept",
    "Body System",
    "Occupation or Discipline",
    "Biomedical Occupation or Discipline",
    "Organization",
    "Group",
    "Professional or Organizational group",
    "Population Group",
    "Family Group",
    "Age Group",
    "Patient or Disabled group",
    "Group Attribute",
    "Intellectual Product",
    "Language",
]


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0.0
    predicted_entities = {}
    golden_entities = {}
    all_syn_err = 0
    all_key_err = 0
    wrong = 0
    all_type_err = 0
    num_extractions = 0
    all_errors = []
    all_correct = []
    for doc_id in extractions:
        extracted_entities_raw = extractions[doc_id]
        #extracted_messages = extractions[doc_id]["messages"]
        # here instead dict as a value, I have a list as a value, the first list is the entities and the second list is the entity types but I need this like a dic
        # or a tuple
        gold_entities = entities_by_pmid[doc_id]
        gold_text = text_by_pmid[doc_id]
        title_abs_gold_entities = sentence_wise(gold_text, gold_entities)
        # For each document, postprocess extracted entities

        (
            extracted_entities,
            key_err,
            syn_err,
            type_err,
        ) = postprocess_entity_outputs(extracted_entities_raw)
        # print("After postprocessing: {}".format(extracted_entities))

        predicted_entities[doc_id] = extracted_entities
        golden_entities[doc_id] = title_abs_gold_entities
        error_sents = set()
        correct_sents = set()
        for idx, _ in extracted_entities.items():
            if idx not in title_abs_gold_entities:
                wrong += 1
            if extracted_entities[idx] and idx in title_abs_gold_entities:
                num_extractions += len(extracted_entities[idx]["entity"])

                cur_correct, correct = strict_match(
                    extracted_entities[idx]["entity"], title_abs_gold_entities[idx]
                )

                if cur_correct < len(extracted_entities[idx]["entity"]):
                    error_sents.add(idx)
                
                if cur_correct == len(extracted_entities[idx]["entity"]):
                    correct_sents.add(idx)

                overall_strict_correct += cur_correct

            # Store correctness by entity type and document type
        all_syn_err += syn_err
        all_key_err += key_err
        all_type_err += type_err

    print(wrong)
    print(num_extractions)
    print(overall_strict_correct)

    overall_strict_prec = overall_strict_correct / num_extractions

    total_gold_extractions = 0

    for idx in entities_by_pmid:
        length = len(entities_by_pmid[idx])
        total_gold_extractions += length

    overall_strict_rec = overall_strict_correct / total_gold_extractions
    
    overall_strict_prec = round(overall_strict_prec, 4)
    overall_strict_rec = round(overall_strict_rec, 4)

    print(overall_strict_prec)
    print(overall_strict_rec)
    print(all_syn_err)
    print(all_key_err)
    print(all_type_err)
    True # 

    return overall_strict_prec, overall_strict_rec


def _is_entity_in_sentence(data_dict, offset):
    if (
        data_dict["offsets"][0][0] >= offset[0]
        and data_dict["offsets"][0][1] <= offset[1]
    ):
        return True
    else:
        return False


def sentence_wise(gold_text, gold_entities):
    # With the offsets from
    abs_gold_entities = {}
    offset_list = gold_text[1]
    idx = 0
    for i, sent_offset in enumerate(offset_list):
        sent_gold_entities = []
        while idx < len(gold_entities) and _is_entity_in_sentence(
            gold_entities[idx], sent_offset
        ):
            sent_gold_entities.append(gold_entities[idx]["text"][0])
            idx += 1

        abs_gold_entities[str(i)] = sent_gold_entities
    return abs_gold_entities


def postprocess_entity_outputs(extracted_entities):
    postprocessed_entities = {}
    key_err = 0
    syn_err = 0
    type_err = 0
    for sent_num, extraction in extracted_entities.items():
        
        if "']" in extraction:
            extraction = extraction.split("']")[0]

        if "')" in extraction:
            extraction = extraction.split("')")[0]
        
        if "')]" in extraction:
            extraction = extraction.split("')]")[0]

        if "</bot>" in extraction:
            extraction = extraction.split("</bot>")[0]
        
        if "<bot>" in extraction:
            extraction = extraction.split("<bot>")[0]
        
        if "\n<bot>" in extraction:
            extraction = extraction.split("\n<bot>")[0]

        if "\n<human>" in extraction:
            extraction = extraction.split("\n<human>")[0]

        if "<human>" in extraction:
            extraction = extraction.split("<human>")[0]

        if "</human>" in extraction:
            extraction = extraction.split("</human>")[0]

        if "Sentence" in extraction:
            extraction = extraction.split("Sentence")[0]

        if "\n['" in extraction:
            extraction = extraction.split("\n['")[0]
        
        if '\n["' in extraction:
            extraction = extraction.split('\n["')[0]
        
        if "\n```" in extraction:
            extraction = extraction.split("\n```")[0]

        extraction = extraction.strip()
        extraction = extraction.rstrip("'")
        extraction = extraction.rstrip(",")
        extraction = extraction.rstrip("',")
        extraction = extraction.strip("\', '")

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


        if "type" in extraction:
            entity_dict = {}
            try:
                extraction = ast.literal_eval(extraction)
                entity_dict["entity"] = [item['name'] for item in extraction['entities']]
                postprocessed_entities[sent_num] = entity_dict
            
            except Exception:
                entity_dict["entity"] = []
                postprocessed_entities[sent_num] = entity_dict

        else:
            try:
                extraction = ast.literal_eval(extraction)
                entity_dict = {}
                # GPT outputs are quite clean, we mostly get lists separated
                # by commas or newlines, so we split on those characters
                # In a few rare cases, no entities of a given type are found
                # "entities" for gpt3.5 (text)
                if "Entity" in extraction:
                    entity_dict["entity"] = extraction["Entity"]
                elif "entity" in extraction:
                    entity_dict["entity"] = extraction["entity"]
                elif "Entities" in extraction:
                    entity_dict["entity"] = extraction["Entities"]
                elif "entities" in extraction:
                    entity_dict["entity"] = extraction["entities"]
                elif "text" in extraction:
                    entity_dict["entity"] = extraction["text"]
                else:
                    for key in extraction:
                        if key in ENTITY_TYPES:
                            if sent_num not in postprocessed_entities:
                                entity_dict["entity"] = []
                            entity_dict["entity"] += extraction[key]

                postprocessed_entities[sent_num] = entity_dict

            except KeyError as e:
                #print(e)
                key_err += 1

            except SyntaxError as e:
                #print("-----")
                syn_err += 1
                # print("--------")
                # print(extraction)
                pass

            except TypeError as e:
                #print(e)
                type_err += 1
                pass 

            except ValueError as e:
                syn_err += 1
    
    return postprocessed_entities, key_err, syn_err, type_err


def strict_match(preds, golds):
    correct = False
    try:
        # TODO: Should we evaluate this? Right now it is only one case
        if isinstance(preds[0], dict):
            # preds = [x["text"] for x in preds]
            return 0, correct
        if isinstance(preds[0], list):
            preds = [y for x in preds for y in x]
        preds = [x.lower() for x in preds]
        golds = [x.lower() for x in golds]
    except Exception:
        pass
    # print(preds)
    # print(golds)
    # if set(preds) == set(golds):
    #     correct = True
    cur_entities = len(set(preds).intersection(set(golds)))

    return cur_entities, correct


def create_subsample_dict(dict1, dict2):
    dict3 = {}
    for each_passage in dict1:
        doc_id = each_passage["pmid"]
        if doc_id in dict2:
            dict3[doc_id] = dict2[doc_id]

    return dict3


def main(output_file):
    print(output_file)
    dataset = load_dataset("bigbio/medmentions")["test"]

    print("Loading dataset...")
    extracted_entities = json.load(open(output_file))
    print(len(extracted_entities))

    entities_by_pmid = {}
    text_by_pmid = {}

    for idx, item in enumerate(dataset):
        pmid = item["pmid"]
        text = item["passages"]
        # title offset
        # text offset
        content = text[0]["text"][0] + " " + text[1]["text"][0]
        item["sent_offsets"] = []
        # Adding in abstract dict but this also includes title offsets
        for start, end in PunktSentenceTokenizer().span_tokenize(content):
            cur_offset = [start, end]
            item["sent_offsets"].append(cur_offset)

        text_by_pmid[pmid] = [content, item["sent_offsets"]]
        entities_by_pmid[pmid] = item["entities"]

    prec, rec = eval_entity_extraction(extracted_entities, text_by_pmid, entities_by_pmid)

    return prec, rec


if __name__ == "__main__":
    main()
