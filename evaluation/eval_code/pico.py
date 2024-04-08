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


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0.0
    all_syn_err = 0
    all_key_err = 0
    all_type_err = 0
    all_val_err = 0
    num_extractions = 0
    predicted_entities = {}
    all_errors = []
    for doc_id in extractions:
        # for each doc
        extracted_entities_raw = extractions[doc_id]
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
            error_sents,
        ) = postprocess_entity_outputs(extracted_entities_raw, sentence_gold_entities)

        predicted_entities[doc_id] = extracted_entities
        # For every entity type, send it to strict_match with all entities of same type in gold
        overall_strict_correct += correct_preds
        all_syn_err += syn_err
        all_key_err += key_err
        all_val_err += val_err
        all_type_err += type_err
        num_extractions += doc_num_extractions

        all_errors += [
            (
                doc_id,
                x,
                extracted_entities[x],
                sentence_gold_entities[x],
                gold_text["text"][
                    gold_text["sent_offsets"][int(x)][0] : gold_text["sent_offsets"][
                        int(x)
                    ][1]
                ],
            )
            for x in error_sents
        ]
        # Store correctness by entity type and document type

    # random.shuffle(all_errors)
    # sampled_errors = all_errors[:100]

    # error_file_path = OUTPUT_DIR / f"/final_errors/pico/best_fs_for_human_eval.txt"
    # with open(
    #     error_file_path,
    #     "a+",
    # ) as f:
    #     f.write("\nDoc start\n")
    #     f.write(str(sampled_errors))

    # print("Saved at ", error_file_path)
    
    
    overall_strict_prec = overall_strict_correct / num_extractions

    total_gold_preds = 0
    for pmid in entities_by_pmid:
        length = len(entities_by_pmid[pmid])
        total_gold_preds += length

    overall_strict_rec = overall_strict_correct / total_gold_preds

    overall_strict_prec = round(overall_strict_prec, 4)
    overall_strict_rec = round(overall_strict_rec, 4)

    print(overall_strict_prec)
    print(overall_strict_rec)
    print(all_syn_err)
    print(all_key_err)
    print(all_type_err)
    print(all_val_err)

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


def transform_list_to_dict(data_list):
    transformed_dict = {
        "population": [],
        "intervention": [],
        "comparator": [],
        "outcome": [],
    }
    error = 0
    try:
        True # 
        for item in data_list:
            text = item["text"]
            item_type = item["type"]

            if item_type == "population":
                transformed_dict["population"].append(text)
            elif item_type == "intervention":
                transformed_dict["intervention"].append(text)
            elif item_type == "comparator":
                transformed_dict["intervention"].append(text)
            elif item_type == "outcome":
                transformed_dict["outcome"].append(text)

    except TypeError as e:
        print(e)
        error += 1
        pass

    except KeyError as e:
        error += 1
        print(e)
        True # 

    return transformed_dict, error


def postprocess_entity_outputs(extracted_entities, gold_extraction):
    """
    This function postprocesses the predicted extractions, gets it back into the json format and also takes the gold extractions and maps them with the predicted.
    The counter keeps the track of number of correct exstrations for calculating precision and recall.
    """
    postprocessed_entities = {}
    processed_extractions = {}
    total_num_extraction = 0
    correct_preds = 0
    syn_err = 0
    key_err = 0
    val_err = 0
    type_err = 0
    error_sents = set()

    for sentence_number, extraction in extracted_entities.items():
        entity_list = []
        
        # skip this in fewshot
        extraction = "entity_list.append({'" + extraction

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


        if "return" in extraction:
            extraction = extraction.replace("return entity_list", "")

        if extraction.endswith("'"):
            extraction = extraction + " '})"

        if extraction.endswith("\n"):
            extraction = extraction[:-2]

        if extraction.endswith("\n "):
            extraction = extraction[:-3]

        if not extraction.endswith("'})"):
            extraction = [x.strip() for x in extraction.split("\n")]
            extraction = "\n".join(extraction[:-1])

        extraction = "\n".join([x.strip() for x in extraction.split("\n")])

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


        try:
            exec(extraction)
        except Exception as e:
            syn_err += 1

            pass

        if entity_list == []:
            transformed_dict = {"text": ""}
        else:
            transformed_dict = entity_list

        processed_extractions, error = transform_list_to_dict(transformed_dict)

        if processed_extractions["comparator"] != []:
            True # 

        compared_indices = set()
        num_extractions = sum([len(v) for k, v in processed_extractions.items()])
        total_num_extraction += num_extractions

        for i in range(len(gold_extraction[sentence_number])):
            text = gold_extraction[sentence_number][i]["text"]
            gold_type = gold_extraction[sentence_number][i]["annotation_type"]

            data_type = ENTITY_TYPES_MAP.get(gold_type)
            if not data_type:
                print(gold_type)

            found = False
            try:
                True # 
                # sentence_number and then retrieving the text and gold - sentence wise
                for j in range(len(processed_extractions[data_type])):
                    if (i, j) not in compared_indices:
                        if processed_extractions[data_type][j] == text:
                            found = True
                            correct_preds += 1
                            compared_indices.add((i, j))
                            compared_indices.add((j, i))
                
                if not found:
                    error_sents.add(sentence_number)

            except KeyError as e:
                # This code will be executed if a KeyError exception is raised
                print(e)
                print(extraction)
                print("This is a Key error")
                True # 
                key_err += 1
                pass

            except SyntaxError as e:
                # This code will be executed if a SyntaxError exception is raised
                print(e)
                print("This is a Syntax error")
                syn_err += 1
                pass

            except ValueError as e:
                # This code will be executed if a SyntaxError exception is raised
                print(e)
                val_err += 1
                print("This is a value error")
                pass

            except TypeError as e:
                # This code will be executed if a SyntaxError exception is raised
                print(e)
                type_err += 1
                print("This is a type error")
                pass

        postprocessed_entities[sentence_number] = processed_extractions

    return (
        postprocessed_entities,
        correct_preds,
        total_num_extraction,
        syn_err,
        key_err,
        val_err,
        type_err,
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
    dataset = load_dataset("bigbio/ebm_pico")["test"]

    print("Loading dataset...")
    extracted_entities = json.load(open(output_file))
    print(output_file)
    # extracted_entities = create_subsample_dict(dataset, all_extracted_entities)
    # print(len(extracted_entities))

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
