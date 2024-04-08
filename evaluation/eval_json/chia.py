import ast
import random
import json
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
    all_val_err = 0
    all_attr_err = 0
    all_type_err = 0
    predicted_entities = {}
    all_errors = []
    all_correct= []
    num_extractions = 0
    for doc_id in extractions:
        # for each doc

        extracted_entities_raw = extractions[doc_id]
        gold_entities = entities_by_pmid[doc_id]
        #extracted_messages = extractions[doc_id]["messages"]
        gold_text = text_by_pmid[doc_id]
        all_sents = text_by_pmid[doc_id]["all_sents"]
        sentence_gold_entities = sentence_wise(gold_text, gold_entities)

        # For each document, postprocess extracted entities
        (
            extracted_entities,
            correct_preds,
            doc_num_extractions,
            key_err,
            syn_err,
            val_err,
            type_err,
            idx_err,
            attr_err,
            error_sents,
            correct_sents, 
        ) = postprocess_entity_outputs(extracted_entities_raw, sentence_gold_entities)

        predicted_entities[doc_id] = extracted_entities
        # For every entity type, send it to strict_match with all entities of same type in gold
        overall_strict_correct += correct_preds
        all_syn_err += syn_err
        all_key_err += key_err
        all_idx_err += idx_err
        all_val_err += val_err
        all_attr_err += attr_err
        all_type_err += type_err
        num_extractions += doc_num_extractions
        # Store correctness by entity type and document type

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
    print(all_val_err)
    print(all_attr_err)
    print(all_type_err)

    return overall_strict_prec, overall_strict_rec


def _is_entity_in_sentence(data_dict, offset):
    if (
        data_dict["offsets"][0][0] >= offset[0] - 1
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

    sorted_list = sorted(gold_entities, key=lambda x: x["offsets"])
    for i, sent_offset in enumerate(offset_list[0]):
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
    syn_err = 0
    key_err = 0
    idx_err = 0
    val_err = 0
    type_err = 0
    idx_err = 0
    attr_err = 0
    total_num_extraction = 0
    error_sents = set()
    correct_sents = set()

    for sentence_number, extraction in extracted_entities.items():
        
        if not extraction.endswith('"]}'):
            extraction = extraction + '"]}'
        
        if extraction.endswith(']}"]}'):
            extraction = extraction[:-3]

        # if '"]' in extraction:
        #     extraction = extraction.split('"]')[0]

        # if "']" in extraction:
        #     extraction = extraction.split("']")[0]

        # if "')" in extraction:
        #     extraction = extraction.split("')")[0]
        
        # if "')]" in extraction:
        #     extraction = extraction.split("')]")[0]
            
        # if "\n<human>" in extraction:
        #     extraction = extraction.split("\n<human>")[0]

        # if "<human>" in extraction:
        #     extraction = extraction.split("<human>")[0]
        
        # if "\'<human>" in extraction:
        #     extraction = extraction.split("\'<human>")[0]

        # if "</human>" in extraction:
        #     extraction = extraction.split("</human>")[0]

        # if "Sentence" in extraction:
        #     extraction = extraction.split("Sentence")[0]

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

        extraction = extraction.strip()


        try:
            extraction = ast.literal_eval(extraction)

            processed_extractions = {
                "value": [],
                "reference_point": [],
                "device": [],
                "multiplier": [],
                "condition": [],
                "temporal": [],
                "person": [],
                "drug": [],
                "negation": [],
                "measurement": [],
                "procedure": [],
                "visit": [],
                "mood": [],
                "qualifier": [],
                "observation": [],
                "scope": [],
            }
            for k, v in extraction.items():
                for chia in processed_extractions.keys():
                    if chia in k.lower():
                        processed_extractions[chia] += v

            num_extractions = sum([len(v) for k, v in processed_extractions.items()])
            total_num_extraction += num_extractions
            compared_indices = set()

            for i in range(len(gold_extraction[sentence_number])):
                text = gold_extraction[sentence_number][i]["text"][0]

                gold_type = gold_extraction[sentence_number][i]["type"]
                data_type = gold_type.lower()

                found = False

                for j in range(len([processed_extractions[data_type]])):
                    if (i, j) not in compared_indices:
                        if processed_extractions[data_type]:
                            if processed_extractions[data_type][j] == text:
                                correct_preds += 1
                                found = True
                                compared_indices.add((i, j))
                                compared_indices.add((j, i))

                if not found:
                    error_sents.add(sentence_number)
                else:
                    correct_sents.add(sentence_number)


            postprocessed_entities[sentence_number] = processed_extractions

        except KeyError as e:
            # This code will be executed if a KeyError exception is raised
            # print(e)
            # print(extraction)
            # print("This is a Key error")
            True # 
            key_err += 1
            pass

        except SyntaxError as e:
            # This code will be executed if a SyntaxError exception is raised
            #print(e)
            #print("This is a Syntax error")

    
            syn_err += 1
            pass

        except ValueError as e:
            # This code will be executed if a SyntaxError exception is raised
            #print(e)
            val_err += 1
            pass

        except AttributeError as e:
            #print(e)
            attr_err += 1
            pass

    return (
        postprocessed_entities,
        correct_preds,
        total_num_extraction,
        key_err,
        syn_err,
        val_err,
        type_err,
        idx_err,
        attr_err,
        error_sents,
        correct_sents, 
    )


def create_subsample_dict(dict1, dict2):
    dict3 = {}
    for each_passage in dict1:
        doc_id = each_passage["id"]
        if doc_id in dict2:
            dict3[doc_id] = dict2[doc_id]

    return dict3

def main(output_file):
    print("Loading dataset...")
    dataset = load_from_disk(
        ""replace with the chia data split path"/test.json"
    )
    
    extracted_entities = json.load(open(output_file))
    print(len(extracted_entities))

    entities_by_pmid = {}
    text_by_pmid = {}

    def split_abstract_into_sentences(abstract):
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
        all_sents, offsets = split_abstract_into_sentences(content)
        cur_offset = offsets
        sent_offsets.append(cur_offset)

        text_by_pmid[pmid] = {}
        text_by_pmid[pmid]["text"] = content
        text_by_pmid[pmid]["all_sents"] = all_sents
        text_by_pmid[pmid]["sent_offsets"] = sent_offsets
        entities_by_pmid[pmid] = example["entities"]

    prec, rec = eval_entity_extraction(extracted_entities, text_by_pmid, entities_by_pmid)

    return prec, rec


if __name__ == "__main__":    
    main()
