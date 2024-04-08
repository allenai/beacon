import re
import io
import ast
import json
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from typing import List, Dict, Any
from collections import defaultdict
from nltk.tokenize.punkt import PunktSentenceTokenizer

ENTITY_TYPES = ["CHEMICAL", "GENE-N", "GENE-Y"]


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0.0
    all_syn_err = 0
    all_key_err = 0
    all_idx_err = 0
    num_extractions = 0
    for doc_id in extractions:
        extracted_entities_raw = extractions[doc_id]
        gold_entities = entities_by_pmid[doc_id]
        gold_text = text_by_pmid[doc_id]
        sentence_gold_entities = sentence_wise(gold_text, gold_entities)
        (
            extracted_entities,
            correct_preds,
            doc_num_extractions,
            syn_err,
            key_err,
            idx_err,
        ) = postprocess_entity_outputs(extracted_entities_raw, sentence_gold_entities)
        extractions[doc_id] = extracted_entities
        overall_strict_correct += correct_preds
        all_syn_err += syn_err
        all_key_err += key_err
        all_idx_err += idx_err
        num_extractions += doc_num_extractions
        # Store correctness by entity type and document type

    True # 
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
    print(overall_strict_prec)
    print(overall_strict_rec)
    print(all_syn_err)
    print(all_key_err)
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
    if (
        data_dict["offsets"][0][0] >= offset[0]
        and data_dict["offsets"][0][1] <= offset[1]
    ):
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

    for i, sent_offset in enumerate(offset_list):
        sent_gold_entities = []
        while idx < len(gold_entities) and _is_entity_in_sentence(
            gold_entities[idx], sent_offset
        ):
            sent_gold_entities.append(gold_entities[idx])
            idx += 1

        abs_gold_entities[str(i)] = sent_gold_entities
    return abs_gold_entities


def transform_list_to_dict(data_list):
    transformed_dict = {"chemical": [], "disease": []}
    error = 0
    try:
        for item in data_list:
            text = item["text"]
            item_type = item["type"]

            if "chemical" in item_type:
                transformed_dict["chemical"].append(text)
            elif "disease" in item_type:
                transformed_dict["disease"].append(text)

    except TypeError as e:
        print(e)
        error += 1
        pass

    except KeyError as e:
        print(e)

    return transformed_dict, error


def postprocess_entity_outputs(extracted_entities, gold_extraction):
    """
    This function postprocesses the predicted extractions, gets it back into the json format and also takes the gold extractions and maps them with the predicted.
    The counter keeps the track of number of correct exstrations for calculating precision and recall.
    """
    postprocessed_entities = {}
    correct_preds = 0
    syn_err = 0
    key_err = 0
    idx_err = 0
    total_num_extraction = 0
    for sentence_number, extraction in extracted_entities.items():
        entity_list = []
        # comment this for few shot
        # if extraction.startswith("'") or extraction.startswith('"'):
        #     extraction = "entity_list.append({" + extraction
        # else:
        #     extraction = "entity_list.append({'" + extraction

        # this is for few shot
        if "entity_list = []" in extraction:
            extraction = extraction.split("entity_list = []")[1]

        # what to do if there is input sentence?
        if "input_text" in extraction:
            extraction = "\n".join([x.strip() for x in extraction.split("\n")][-1:])

        if "```python" in extraction:
            extraction = extraction.split("```python")[1]

        # this is mostly for gpt OUTPUTS
        if "'}) " in extraction:
            extraction = extraction.replace("'}) ", "'})\n")

        # also for GPT :
        if extraction.startswith("entity_list.append()"):
            extraction = extraction.split("entity_list.append()")[1]
            extraction = extraction.strip()

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

        pattern = r"(?<=\w)\'(?=\w)"
        # Define the replacement string
        replacement = r"\\"
        # Use re.sub() to replace the matched patterns with the replacement
        extraction = re.sub(pattern, replacement, extraction)

        try:
            exec(extraction)
        except:
            print("-----")
            print(extraction)

            syn_err += 1
            pass

        if entity_list == []:
            transformed_dict = {"text": ""}
        else:
            transformed_dict = entity_list

        processed_extractions, error = transform_list_to_dict(transformed_dict)
        try:
            compared_indices = set()
            num_extractions = sum([len(v) for k, v in processed_extractions.items()])
            total_num_extraction += num_extractions
            compared_indices = set()

            for i in range(len(gold_extraction[sentence_number])):
                text = gold_extraction[sentence_number][i]["text"][0]
                gold_type = gold_extraction[sentence_number][i]["type"]
                data_type = gold_type.lower()

                for j in range(len(processed_extractions[data_type])):
                    if (i, j) not in compared_indices:
                        if processed_extractions[data_type][j] == text:
                            correct_preds += 1
                            compared_indices.add((i, j))
                            compared_indices.add((j, i))

            postprocessed_entities[sentence_number] = processed_extractions

        except KeyError as e:
            # This code will be executed if a SyntaxError exception is raised
            print(e)
            print(extraction)
            print("This is a Key error")
            key_err += 1
            pass

        except SyntaxError as e:
            print(e)
            print(extraction)
            print("This is a syntax error")
            syn_err += 1
            pass

        except IndexError as e:
            print(e)
            print("This is an index error")
            idx_err += 1
            pass

    return (
        postprocessed_entities,
        correct_preds,
        total_num_extraction,
        syn_err,
        key_err,
        idx_err,
    )

def main(output_file):
    print("Loading dataset...")
    dataset = load_dataset("bigbio/bc5cdr")["test"]
    extracted_entities = json.load(open(output_file))

    entities_by_pmid = {}
    text_by_pmid = {}

    for idx, example in enumerate(dataset):
        entities = []
        pmid = example["passages"][0]["document_id"]
        text = example["passages"]
        # title offset
        # text offset

        # add space in here for cdr code
        content = text[0]["text"] + " " + text[1]["text"]
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
    