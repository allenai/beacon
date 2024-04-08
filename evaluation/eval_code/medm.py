import re
import io
import ast
import json
import nltk
from ipdb import set_trace
from datasets import load_dataset
from typing import List, Dict, Any
from collections import defaultdict
import collections
import warnings
from itertools import chain
from nltk.tokenize.punkt import PunktSentenceTokenizer

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
    all_type_err = 0
    num_extractions = 0
    wrong = 0
    for doc_id in extractions:
        syn_err = 0
        key_err = 0
        type_err = 0
        extracted_entities_raw = extractions[doc_id]
        # here instead dict as a value, I have a list as a value, the first list is the entities and the second list is the entity types but I need this like a dic
        # or a tuple
        gold_entities = entities_by_pmid[doc_id]
        gold_text = text_by_pmid[doc_id]
        title_abs_gold_entities = sentence_wise(gold_text, gold_entities)

        # For each document, postprocess extracted entities
        extracted_entities, syn_err, key_err, type_err = postprocess_entity_outputs(
            extracted_entities_raw
        )
        predicted_entities[doc_id] = extracted_entities
        golden_entities[doc_id] = title_abs_gold_entities
        for idx, _ in extracted_entities.items():
            if idx not in title_abs_gold_entities:
                wrong += 1
            if extracted_entities[idx] and idx in title_abs_gold_entities:
                num_extractions += len(extracted_entities[idx]["entity"])

                cur_correct = strict_match(
                    extracted_entities[idx]["entity"], title_abs_gold_entities[idx]
                )

                overall_strict_correct += cur_correct
            # Store correctness by entity type and document type
            # edit precision and recall
        all_syn_err += syn_err
        all_key_err += key_err
        all_type_err += type_err

    True # 
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
    offset_list = gold_text[1]["sent_offsets"]
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


def transform_list_to_dict(entity_list):
    transformed_dict = {"entity": []}
    error = 0

    if entity_list:
        for i in range(len(entity_list)):
            if "Entity" in entity_list[i]:
                if isinstance(entity_list[i]["Entity"], list):
                    transformed_dict["entity"] += entity_list[i]["Entity"]
                else:
                    transformed_dict["entity"].append(entity_list[i]["Entity"])
            elif "entity" in entity_list[i]:
                if isinstance(entity_list[i]["entity"], list):
                    transformed_dict["entity"] += entity_list[i]["entity"]
                else:
                    transformed_dict["entity"].append(entity_list[i]["entity"])
            elif "Entities" in entity_list[i]:
                if isinstance(entity_list[i]["Entities"], list):
                    transformed_dict["entity"] += entity_list[i]["Entities"]
                else:
                    transformed_dict["entity"].append(entity_list[i]["Entities"])
            elif "entities" in entity_list[i]:
                if isinstance(entity_list[i]["entities"], list):
                    transformed_dict["entity"] += entity_list[i]["entities"]
                else:
                    transformed_dict["entity"].append(entity_list[i]["entities"])
            elif "text" in entity_list[i]:
                if isinstance(entity_list[i]["text"], list):
                    transformed_dict["entity"] += entity_list[i]["text"]
                else:
                    transformed_dict["entity"].append(entity_list[i]["text"])

            else:
                for key in entity_list[i]:
                    if key in ENTITY_TYPES:
                        transformed_dict["entity"] += entity_list[i][key]

    else:
        print("Empty dict")

    return transformed_dict, error


def postprocess_entity_outputs(extracted_entities):
    postprocessed_entities = {}
    syn_err = 0
    key_err = 0
    type_err = 0
    for sent_num, extractions in extracted_entities.items():
        entity_list = []
        # comment this in fewshot
        #extractions = "entity_list.append({'" + extractions

        if extractions.startswith('entity_list.append({"entities": )'):
            extractions = extractions.replace('entity_list.append({"entities": )', "")

        if "return" in extractions:
            extractions = extractions.replace("return entity_list", "")

        if extractions.endswith("'"):
            extractions = extractions + " '})"

        if "'}) " in extractions:
            extractions = extractions.replace("'}) ", "'})\n")

        if extractions.endswith("\n"):
            extractions = extractions[:-2]

        if extractions.endswith("\n "):
            extractions = extractions[:-3]

        if not extractions.endswith("'})"):
            extractions = [x.strip() for x in extractions.split("\n")]
            extractions = "\n".join(extractions[:-1])

        extractions = "\n".join([x.strip() for x in extractions.split("\n")])

        try:
            exec(extractions)
        except:
            syn_err += 1
            print(extractions)
            pass

        processed_extractions, error = transform_list_to_dict(entity_list)
        postprocessed_entities[sent_num] = processed_extractions

    return postprocessed_entities, syn_err, key_err, type_err


def strict_match(preds, golds):
    try:
        preds = [x.lower() for x in preds]
        golds = [x.lower() for x in golds]
    except Exception:
        pass
    # print(preds)
    # print(golds)

    return len(set(preds).intersection(set(golds)))

def main(output_file):
    print("Loading dataset...")
    dataset = load_dataset("bigbio/medmentions")["test"]
    extracted_entities = json.load(open(output_file))

    entities_by_pmid = {}
    text_by_pmid = {}

    for idx, example in enumerate(dataset):
        pmid = example["pmid"]
        text = example["passages"]
        # title offset
        # text offset
        # add a space in fewshot
        content = text[0]["text"][0] + " " + text[1]["text"][0]
        text[1]["sent_offsets"] = []
        # Adding in abstract dict but this also includes title offsets
        for start, end in PunktSentenceTokenizer().span_tokenize(content):
            cur_offset = [start, end]
            text[1]["sent_offsets"].append(cur_offset)

        text_by_pmid[pmid] = text
        entities_by_pmid[pmid] = example["entities"]

    prec, rec = eval_entity_extraction(extracted_entities, text_by_pmid, entities_by_pmid)
    return prec, rec



if __name__ == "__main__":
    main()

    