import ast
import json
import random
from ipdb import set_trace
from datasets import load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0.0
    all_syn_err = 0.0
    all_key_err = 0
    all_idx_err = 0
    all_err = 0
    predicted_entities = {}
    num_extractions = 0
    all_errors = []
    for doc_id in extractions:
        # for each doc
        extracted_entities_raw = extractions[doc_id]
        gold_entities = entities_by_pmid[doc_id]
        gold_text = text_by_pmid[doc_id]
        all_sents = text_by_pmid[doc_id]["all_sents"]
        sentence_gold_entities = sentence_wise(gold_text, gold_entities)

        # For each document, postprocess extracted entities
        (
            extracted_entities,
            correct_preds,
            doc_num_extractions,
            syn_err,
            key_err,
            idx_err,
            error,
            error_sents,
        ) = postprocess_entity_outputs(extracted_entities_raw, sentence_gold_entities)

        predicted_entities[doc_id] = extracted_entities
        # For every entity type, send it to strict_match with all entities of same type in gold
        overall_strict_correct += correct_preds
        all_syn_err += syn_err
        all_key_err += key_err
        all_idx_err += idx_err
        all_err += error
        num_extractions += doc_num_extractions
        # Store correctness by entity type and document type
        # -2 here is to compensate for the added + 2 in the offsets code for "\n"

        all_errors += [
            (
                doc_id,
                x,
                extracted_entities[x],
                sentence_gold_entities[x],
                all_sents[int(x)],
            )
            for x in error_sents
        ]

    print(overall_strict_correct)
    print(num_extractions)

    # random.shuffle(all_errors)
    # sampled_errors = all_errors[:100]

    # error_file_path = OUTPUT_DIR / f"/final_errors/chia/best_fs_for_human_eval.txt"
    # with open(
    #     error_file_path,
    #     "a+",
    # ) as f:
    #     f.write("\nDoc start\n")
    #     f.write(str(sampled_errors))

    # print("Saved at ", error_file_path)

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
    print(all_idx_err)
    print(error)

    return overall_strict_prec, overall_strict_rec


def _is_entity_in_sentence(data_dict, offset):
    if (
        data_dict["offsets"][0][0] >= offset[0] - 1
        and data_dict["offsets"][0][1] <= offset[1] + 1
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

    sorted_list = sorted(gold_entities, key=lambda x: x["offsets"])
    for i, sent_offset in enumerate(offset_list[0]):
        sent_gold_entities = []
        while idx < len(sorted_list) and _is_entity_in_sentence(
            sorted_list[idx], sent_offset
        ):
            sent_gold_entities.append(sorted_list[idx])
            idx += 1

        abs_gold_entities[str(i)] = sent_gold_entities

    True # 
    return abs_gold_entities


def transform_list_to_dict(data_list):
    """
    Condition, Device, Drug, Measurement, Mood, Multiplier, Negation, Observation,
    Person, Procedure, Qualifier, Reference_point, Scope, Temporal, Value, Visit
    """
    error = 0
    transformed_dict = {
        "condition": [],
        "device": [],
        "drug": [],
        "measurement": [],
        "mood": [],
        "multiplier": [],
        "negation": [],
        "observation": [],
        "person": [],
        "procedure": [],
        "qualifier": [],
        "reference_point": [],
        "scope": [],
        "temporal": [],
        "value": [],
        "visit": [],
    }
    try:
        for item in data_list:
            text = item["text"]
            item_type = item["type"]
            if text:
                if item_type.lower() == "condition":
                    True # 
                    transformed_dict["condition"].append(text)
                elif item_type.lower() == "device":
                    transformed_dict["device"].append(text)
                elif item_type.lower() == "drug":
                    transformed_dict["drug"].append(text)
                elif item_type.lower() == "measurement":
                    transformed_dict["measurement"].append(text)
                elif item_type.lower() == "device":
                    transformed_dict["device"].append(text)
                elif item_type.lower() == "mood":
                    transformed_dict["mood"].append(text)
                elif item_type.lower() == "multiplier":
                    transformed_dict["multiplier"].append(text)
                elif item_type.lower() == "negation":
                    transformed_dict["negation"].append(text)
                elif item_type.lower() == "observation":
                    transformed_dict["observation"].append(text)
                elif item_type.lower() == "person":
                    transformed_dict["person"].append(text)
                elif item_type.lower() == "procedure":
                    transformed_dict["procedure"].append(text)
                elif item_type.lower() == "qualifier":
                    transformed_dict["qualifier"].append(text)
                elif item_type.lower() == "reference_point":
                    transformed_dict["reference_point"].append(text)
                elif item_type.lower() == "scope":
                    transformed_dict["scope"].append(text)
                elif item_type.lower() == "temporal":
                    transformed_dict["temporal"].append(text)
                elif item_type.lower() == "value":
                    transformed_dict["value"].append(text)
                elif item_type.lower() == "visit":
                    transformed_dict["visit"].append(text)

    except TypeError as e:
        # print(e)
        error += 1

        pass

    except KeyError as e:
        # print(e)
        pass

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
    error = 0
    error_sents = set()
    total_num_extraction = 0
    for sentence_number, extraction in extracted_entities.items():
        entity_list = []

        # for few shot comment this
        if extraction.startswith("entity_list.append({'"):
            pass
        else:
            if extraction.startswith("'") or extraction.startswith('"'):
                extraction = "entity_list.append({" + extraction
            else:
                extraction = "entity_list.append({'" + extraction


        if "\n<human>" in extraction:
            extraction = extraction.split("\n<human>")[0]

        if "<human>" in extraction:
            extraction = extraction.split("<human>")[0]
        
        if "\'<human>" in extraction:
            extraction = extraction.split("\'<human>")[0]

        if "</human>" in extraction:
            extraction = extraction.split("</human>")[0]

        if "Sentence" in extraction:
            extraction = extraction.split("Sentence")[0]

        if "\']\n" in extraction:
            extraction = extraction.split("\']\n")[0]

        if "\n['" in extraction:
            extraction = extraction.split("\n['")[0]
        
        if '\n["' in extraction:
            extraction = extraction.split('\n["')[0]
        
        if "\n```" in extraction:
            extraction = extraction.split("\n```")[0]
        
        if "'\n```" in extraction:
            extraction = extraction.split("'\n```")[0]
            

        if "return" in extraction:
            extraction = extraction.replace("return entity_list", "")

        if extraction.endswith("'"):
            extraction = extraction + " })"

        if not extraction.endswith("'})"):
            extraction = [x.strip() for x in extraction.split("\n")]
            extraction = "\n".join(extraction[:-1])

        extraction = "\n".join([x.strip() for x in extraction.split("\n")])

        try:
            exec(extraction)

        except:
            syn_err += 1

            pass

        if entity_list:
            processed_extractions, error = transform_list_to_dict(entity_list)
            num_extractions = sum([len(v) for k, v in processed_extractions.items()])
            total_num_extraction += num_extractions
            compared_indices = set()
            for i in range(len(gold_extraction[sentence_number])):
                text = gold_extraction[sentence_number][i]["text"][0]
                gold_type = gold_extraction[sentence_number][i]["type"]
                data_type = gold_type.lower()
                found = False
                try:
                    for j in range(len(processed_extractions[data_type])):
                        if (i, j) not in compared_indices:
                            if (
                                processed_extractions[data_type][j]
                                == text
                            ):
                                correct_preds += 1
                                found = True
                                compared_indices.add((i, j))
                                compared_indices.add((j, i))

                    if not found:
                        error_sents.add(sentence_number)

                except KeyError as e:
                    # print(e)
                    # print(extraction)
                    # print("This is a Key error")
                    key_err += 1
                    pass

                except SyntaxError as e:
                    # print(e)
                    # print("This is a Syntax error")
                    syn_err += 1
                    pass

                except ValueError as e:
                    # print(e)
                    # print("This is a value error")
                    pass

                except TypeError as e:
                    # print(e)
                    # print("This is a type error")
                    pass

                except IndexError as e:
                    # print(e)
                    idx_err += 1
                    pass

                except AttributeError as e:
                    # print(e)
                    pass

                postprocessed_entities[sentence_number] = processed_extractions

    return (
        postprocessed_entities,
        correct_preds,
        total_num_extraction,
        syn_err,
        key_err,
        idx_err,
        error,
        error_sents,
    )


def create_subsample_dict(dict1, dict2):
    dict3 = {}
    for each_passage in dict1:
        doc_id = each_passage["id"]
        if doc_id in dict2:
            dict3[doc_id] = dict2[doc_id]

    return dict3


def main(output_file):
    
    dataset = load_from_disk(
        ""replace with the chia data split path"/test.hf"
    )    
    print("Loading dataset...")
    extracted_entities = json.load(open(output_file))

    # extracted_entities = create_subsample_dict(dataset, all_extracted_entities)
    # print(len(extracted_entities))

    entities_by_pmid = {}
    text_by_pmid = {}

    def split_abstract_into_sentences(pmid, abstract):
        # Split the abstract into sentences using NLTK's sent_tokenize function
        sentences = filter(None, abstract.split("\n"))
        # Compute the sentence offsets
        offsets = []
        start = 0
        all_sents = []
        for sentence in sentences:
            end = start + len(sentence)
            offsets.append((start, end))
            start = end + 2
            all_sents.append(sentence)
        return all_sents, offsets

    for idx, example in enumerate(dataset):
        entities = []
        pmid = example["id"]
        content = example["text"]
        sent_offsets = []
        # Adding in abstract dict, doesn't include title
        all_sents, offsets = split_abstract_into_sentences(pmid, content)
        sent_offsets.append(offsets)

        text_by_pmid[pmid] = {}
        text_by_pmid[pmid]["text"] = content
        text_by_pmid[pmid]["all_sents"] = all_sents
        text_by_pmid[pmid]["sent_offsets"] = sent_offsets
        entities_by_pmid[pmid] = example["entities"]

    prec, rec = eval_entity_extraction(extracted_entities, text_by_pmid, entities_by_pmid)

    return prec, rec


if __name__ == "__main__":
    main()
