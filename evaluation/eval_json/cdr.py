import sys
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


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0
    all_syntax_err = 0
    predicted_entities = {}
    all_key_err = 0
    all_idx_err = 0
    all_check = 0 
    num_extractions = 0
    all_errors = []
    all_correct = []
    for doc_id in extractions:
        extracted_entities_raw = extractions[doc_id]
        #extracted_messages = extractions[doc_id]["messages"]
        gold_entities = entities_by_pmid[doc_id]
        gold_text = text_by_pmid[doc_id]
        sentence_gold_entities = sentence_wise(gold_text, gold_entities)

        (
            extracted_entities,
            correct_preds,
            doc_num_extraction,
            syntax_err,
            key_err,
            index_err,
            error_sents,
            correct_sents, 
            check,
        ) = postprocess_entity_outputs(
            extracted_entities_raw,
            sentence_gold_entities,
        )
        predicted_entities[doc_id] = extracted_entities
        overall_strict_correct += correct_preds
        all_syntax_err += syntax_err
        all_key_err += key_err
        all_idx_err += index_err
        num_extractions += doc_num_extraction
        all_check += check


    overall_strict_prec = overall_strict_correct / num_extractions

    total_gold_extractions = 0
    for idx in entities_by_pmid:
        length = len(entities_by_pmid[idx])
        total_gold_extractions += length

    overall_strict_rec = overall_strict_correct / total_gold_extractions
    
    overall_strict_prec = round(overall_strict_prec, 4)
    overall_strict_rec = round(overall_strict_rec, 4)

    print(num_extractions)
    print(overall_strict_correct)
    print(total_gold_extractions)
    print("Precision and Recall")
    print(overall_strict_prec)
    print(overall_strict_rec)
    print(all_syntax_err)
    print(all_key_err)
    print(all_idx_err)
    print(all_check)


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
    """
    This function works with gold extrations and looks at the sentence offsets saved in gold_text and buckets the entities from gold_entities into sentences.
    """
    abs_gold_entities = {}
    offset_list = gold_text["sent_offsets"]
    idx = 0

    for i, sent_offset in enumerate(offset_list):
        sent_gold_entities = []
        while idx < len(gold_entities) and _is_entity_in_sentence(
            gold_entities[idx], sent_offset
        ):
            sent_gold_entities.append(gold_entities[idx])
            idx += 1

        abs_gold_entities[str(i)] = sent_gold_entities
    return abs_gold_entities

def combine_entity_extractions(extractions):
    output = []
    for ent_type, ent_text_list in extractions.items():
            output += ent_text_list
    return output


def postprocess_entity_outputs(extracted_entities, gold_extraction):
    """
    This function postprocesses the predicted extractions, gets it back into the json format and also takes the gold extractions and maps them with the predicted.
    The counter keeps the track of number of correct exstrations for calculating precision and recall.
    """
    postprocessed_entities = {}
    correct_preds = 0
    syntax_err = 0
    key_err = 0
    check = 0 
    index_err = 0
    total_num_extraction = 0
    error_sents = set()
    correct_sents = set()
    for sentence_number, extraction in extracted_entities.items():
        
        # if "']" in extraction:
        #     extraction = extraction.split()
        
        if "')" in extraction:
            extraction = extraction.split("')")[0]

        if "</bot>" in extraction:
            extraction = extraction.split("</bot>")[0]

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

        extraction = extraction.strip()
        extraction = extraction.strip("']")
        extraction = extraction.rstrip("',")
        extraction = extraction.strip("\', '")

        # if "[" == extraction[0]:
        #     extraction = extraction.lstrip("[")
        
        # if "[" == extraction[-1]:
        #     extraction = extraction.rstrip("]")


        if "Here " in extraction:
            try:
                if "format:" in extraction:
                    extraction = extraction.split("format:")[1]
                elif "extracted:" in extraction:
                    extraction = extraction.split("extracted:")[1]
                
            except IndexError as e:
                True # 

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
                        check += 1 
                        raise ValueError("One of the inputs is not a valid dictionary.")

                except SyntaxError as e:
                    
                    syntax_err += 1
                    # print("--------")
                    # print(extraction)
                    # with open(
                    #     OUTPUT_DIR / f"/error_analysis/cdr/seed12_{kshot}_rerun.txt",
                    #     "a+",
                    # ) as f:
                    #     f.write("\nDoc start\n")
                    #     f.write(str(dict_str))
                    pass

                except ValueError as e:
                    pass

                single_dict = single_dict_fn(dictionaries)
                extraction = flatten_lists_in_dict(single_dict)

        except ValueError as e:
            pass
        
        try:
  
            processed_extractions = {"chemical": [], "disease": []}
            if not isinstance(extraction, dict):
                # print(extraction)
                continue
            for k, v in extraction.items():
                if "chemical" in k.lower():
                    processed_extractions["chemical"] += v
                if "disease" in k.lower():
                    processed_extractions["disease"] += v

            num_extractions = sum([len(v) for k, v in processed_extractions.items()])
            total_num_extraction += num_extractions
            compared_indices = set()

            combined_extractions = combine_entity_extractions(processed_extractions)
            is_gold_extractions_equal_len = len(gold_extraction[sentence_number]) == len(combined_extractions)

            found = False
            num_found = 0

            for i in range(len(gold_extraction[sentence_number])):
                text = gold_extraction[sentence_number][i]["text"][0]
                gold_type = gold_extraction[sentence_number][i]["type"]
                data_type = gold_type.lower()

                # if text in combined_extractions and text in processed_extractions[data_type]:
                #     correct_preds += 1
                #     found = True
                #     num_found += 1

                for j in range(len(processed_extractions[data_type])):
                    if (i, j) not in compared_indices:
                        # change this .lower()
                        if processed_extractions[data_type][j] == text:
                            correct_preds += 1
                            found = True
                            num_found += 1

                            compared_indices.add((i, j))
                            compared_indices.add((j, i))

                if not found:
                    error_sents.add(sentence_number)
                elif is_gold_extractions_equal_len and num_found == len(gold_extraction[sentence_number]):
                    correct_sents.add(sentence_number)

            postprocessed_entities[sentence_number] = processed_extractions

        # Convert the string to a list of dictionaries

        except KeyError as e:
            # This code will be executed if a SyntaxError exception is raised
            # print(e)
            key_err += 1
            pass

        except IndexError as e:
            # print(e)
            index_err += 1
            pass

        except ValueError as e:
            # print(e)
            pass

    return (
        postprocessed_entities,
        correct_preds,
        total_num_extraction,
        syntax_err,
        key_err,
        index_err,
        error_sents,
        correct_sents,
        check,
    )


def count_repeated_elements(input_list):
    element_count = {}
    for element in input_list:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1

    return element_count


def flatten_lists_in_dict(input_dict):
    flattened_dict = {}
    if input_dict:
        for key, value in input_dict.items():
            flattened_list = []
            for sublist in value:
                flattened_list.extend(sublist)

            flattened_dict[key] = flattened_list
            count_repeated_elements(flattened_list)

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
        doc_id = each_passage["passages"][0]["document_id"]
        if doc_id in dict2:
            dict3[doc_id] = dict2[doc_id]

    return dict3


def main(output_file):
    # This is for subsampled val

    print("Loading dataset...")
    print(output_file)
    dataset = load_dataset("bigbio/bc5cdr")["test"]
    extracted_entities = json.load(open(output_file))
    print(len(extracted_entities))

    entities_by_pmid = {}
    text_by_pmid = {}

    for idx, example in enumerate(dataset):
        entities = []
        pmid = example["passages"][0]["document_id"]
        text = example["passages"]
        # title offset
        # text offset
        content = text[0]["text"] +  " " + text[1]["text"]
        sent_offsets = []
        # Adding in abstract dict but this also includes title offsets
        for start, end in PunktSentenceTokenizer().span_tokenize(content):
            cur_offset = [start, end]
            sent_offsets.append(cur_offset)

        text_by_pmid[pmid] = {}
        text_by_pmid[pmid]["text"] = content
        text_by_pmid[pmid]["sent_offsets"] = sent_offsets
        entities_by_pmid[pmid] = (
            example["passages"][0]["entities"] + example["passages"][1]["entities"]
        )

    prec, rec = eval_entity_extraction(extracted_entities, text_by_pmid, entities_by_pmid)

    return prec, rec

if __name__ == "__main__":
    main()
