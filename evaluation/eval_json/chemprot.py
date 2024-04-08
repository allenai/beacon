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

ENTITY_TYPES_MAP = {"CHEMICAL": "chemical", "GENE-N": "protein", "GENE-Y": "protein"}


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0.0
    all_key_err = 0
    all_syn_err = 0
    all_attr_err = 0
    all_idx_err = 0
    predicted_entities = {}
    num_extractions = 0
    all_errors = []
    for doc_id in extractions:
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
            syntax_err,
            key_err,
            attr_err,
            idx_err,
            error_sents,
        ) = postprocess_entity_outputs(
            extracted_entities_raw, sentence_gold_entities, doc_id
        )

        predicted_entities[doc_id] = extracted_entities
        overall_strict_correct += correct_preds
        num_extractions += doc_num_extractions
        all_syn_err += syntax_err
        all_key_err += key_err
        all_attr_err += attr_err
        all_idx_err += idx_err



    overall_strict_prec = overall_strict_correct / num_extractions

    total_gold_extractions = 0
    for idx in entities_by_pmid:
        length = len(entities_by_pmid[idx]["text"])
        total_gold_extractions += length

    overall_strict_rec = overall_strict_correct / total_gold_extractions

    overall_strict_prec = round(overall_strict_prec, 4)
    overall_strict_rec = round(overall_strict_rec, 4)

    print(overall_strict_prec)
    print(overall_strict_rec)
    print(all_syn_err)
    print(all_key_err)
    print(all_attr_err)
    print(all_idx_err)

    return overall_strict_prec, overall_strict_rec


def convert_dict_to_list_of_dicts(dictionary):
    keys = dictionary.keys()
    values = dictionary.values()

    # Get the length of the lists
    length = len(next(iter(values)))

    # Create a list of dictionaries
    list_of_dicts = [{key: dictionary[key][i] for key in keys} for i in range(length)]

    return list_of_dicts


def _is_entity_in_sentence(data_dict, offset):
    if data_dict["offsets"][0] >= offset[0] and data_dict["offsets"][1] <= offset[1]:
        return True
    else:
        return False


def sort_dictionary(dictionary):
    # Get the offsets and sort them
    offsets = dictionary["offsets"]
    sorted_offsets = sorted(offsets)

    # Sort the values based on the sorted offsets
    sorted_values = [value for _, value in sorted(zip(offsets, dictionary.values()))]

    # Create a new sorted dictionary
    sorted_dictionary = {
        "id": sorted_values[0],
        "type": sorted_values[1],
        "text": sorted_values[2],
        "offsets": sorted_offsets,
    }

    return sorted_dictionary


def sentence_wise(gold_text, gold_entities):
    """
    This function works with gold extrations and looks at the sentence offsets saved in gold_text and buckets the entities from gold_entities into sentences.
    """
    abs_gold_entities = {}
    offset_list = gold_text["sent_offsets"]
    idx = 0
    gold_entities_list = convert_dict_to_list_of_dicts(gold_entities)
    sorted_list = sorted(gold_entities_list, key=lambda x: x["offsets"])
    for i, sent_offset in enumerate(offset_list):
        sent_gold_entities = []

        while idx < len(sorted_list) and _is_entity_in_sentence(
            sorted_list[idx], sent_offset
        ):
            sent_gold_entities.append(sorted_list[idx])
            idx += 1

        abs_gold_entities[str(i)] = sent_gold_entities

    return abs_gold_entities


def postprocess_entity_outputs(extracted_entities, gold_extraction, doc_id):
    """
    This function postprocesses the predicted extractions, gets it back into the json format and also takes the gold extractions and maps them with the predicted.
    The counter keeps the track of number of correct exstrations for calculating precision and recall.
    """
    postprocessed_entities = {}
    total_num_extractions = 0
    correct_preds = 0
    syntax_err = 0
    key_err = 0
    attr_err = 0
    idx_err = 0
    error_sents = set()
    for sentence_number, extraction in extracted_entities.items():
        
        if "')" in extraction:
            extraction = extraction.split("')")[0]

        if "', '" in extraction:
            extraction = extraction.split("', '")[0]

        if "</bot>" in extraction:
            extraction = extraction.split("</bot>")[0]

        if "\']\n" in extraction:
            extraction = extraction.split("\']\n")[0]

        if "\')]\n" in extraction:
            extraction = extraction.split("\')]\n")[0]

        if "\n<human>" in extraction:
            extraction = extraction.split("\n<human>")[0]

        if "<human>" in extraction:
            extraction = extraction.split("<human>")[0]

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

        extraction = extraction.strip()
        extraction = extraction.rstrip("'")
        extraction = extraction.rstrip(",")
        extraction = extraction.rstrip("',")
        extraction = extraction.strip("\', '")

    
        try:
            extraction = ast.literal_eval(extraction)

        except SyntaxError as e:
            if extraction:
                dict_list = extraction.split(
                    "\n"
                )  # Replace '\n' with the appropriate delimiter

                dictionaries = []
                dict_str = dict_list[-1]
                try:
                    if dict_str:
                        dictionary = ast.literal_eval(dict_str)
                        if isinstance(dictionary, dict):
                            dictionaries.append(dictionary)

                    else:
                        raise ValueError("One of the inputs is not a valid dictionary.")

                except SyntaxError as e:
                    syntax_err += 1
                    pass

                except ValueError as e:
                    pass

                single_dict = single_dict_fn(dictionaries)
                extraction = flatten_lists_in_dict(single_dict)

        except ValueError as e:
            pass

        try:
            processed_extractions = {"chemical": [], "protein": []}
            for k, v in extraction.items():
                if "chemical" in k.lower() or "chemicals" in k.lower():
                    processed_extractions["chemical"] += v
                if "protein"  in k.lower() or "gene" in k.lower() or "proteins" in k.lower():
                    processed_extractions["protein"] += v

            num_extractions = sum([len(v) for k, v in processed_extractions.items()])
            total_num_extractions += num_extractions
            compared_indices = set()

            for i in range(len(gold_extraction[sentence_number])):
                text = gold_extraction[sentence_number][i]["text"]

                gold_type = gold_extraction[sentence_number][i]["type"]
                data_type = ENTITY_TYPES_MAP.get(gold_type)
                if not data_type:
                    print(gold_type)

                found = False
                for j in range(len(processed_extractions[data_type])):
                    if (i, j) not in compared_indices:
                        if processed_extractions[data_type][j] == text:
                            correct_preds += 1
                            found = True
                            compared_indices.add((i, j))
                            compared_indices.add((j, i))

                if not found:
                    error_sents.add(sentence_number)

            postprocessed_entities[sentence_number] = processed_extractions

        except SyntaxError as e:
            # This code will be executed if a SyntaxError exception is raised
            # print("--------")
            # print(extraction)
            syntax_err += 1

        except KeyError as e:
            key_err += 1
            pass

        except AttributeError as e:
            # print(e)
            attr_err += 1
            pass

        except IndexError as e:
            # print(e)
            idx_err += 1
            pass

        except TypeError as e:
            # True # 
            pass

        except ValueError as e:
            pass
    

    return (
        postprocessed_entities,
        correct_preds,
        total_num_extractions,
        syntax_err,
        key_err,
        attr_err,
        idx_err,
        error_sents,
    )


def flatten_lists_in_dict(input_dict):
    flattened_dict = {}
    if input_dict:
        for key, value in input_dict.items():
            flattened_list = []
            for sublist in value:
                flattened_list.extend(sublist)

            flattened_dict[key] = flattened_list

    return flattened_dict


def single_dict_fn(dict_list):
    # Merge dictionaries into a single dictionary with appended values
    merged_dict = {}

    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    return merged_dict


def create_subsample_dict(dict1, dict2):
    dict3 = {}
    for each_passage in dict1:
        doc_id = each_passage["pmid"]
        if doc_id in dict2:
            dict3[doc_id] = dict2[doc_id]

    return dict3


def main(output_file):
    print(output_file)
    dataset = load_dataset("bigbio/chemprot")["test"]
    print("Loading dataset...")
    extracted_entities = json.load(open(output_file))
    print(len(extracted_entities))

    entities_by_pmid = {}
    text_by_pmid = {}

    for idx, example in enumerate(dataset):
        pmid = example["pmid"]

        content = example["text"]
        # title offset
        sent_offsets = []
        # Adding in abstract dict but this also includes title offsets
        for start, end in PunktSentenceTokenizer().span_tokenize(content):
            cur_offset = [start, end]
            sent_offsets.append(cur_offset)

        text_by_pmid[pmid] = {}
        text_by_pmid[pmid]["text"] = content
        text_by_pmid[pmid]["sent_offsets"] = sent_offsets
        entities_by_pmid[pmid] = example["entities"]

    prec, rec = eval_entity_extraction(extracted_entities, text_by_pmid, entities_by_pmid)

    return prec, rec


if __name__ == "__main__":
    main()
