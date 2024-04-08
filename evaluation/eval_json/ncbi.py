import ast
import json
import re
import random
from ipdb import set_trace
from datasets import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0.0
    predicted_entities = {}
    golden_entities = {}
    all_syn_err = 0
    all_key_err = 0
    all_val_err = 0
    wrong = 0
    cur_correct = 0
    all_type_err = 0
    all_attr_err = 0
    num_extractions = 0
    all_errors = []
    all_correct = []
    for doc_id in extractions:
        key_err = 0
        val_err = 0
        attr_err = 0
        type_err = 0
        extracted_entities_raw = extractions[doc_id]
        #extracted_messages = extractions[doc_id]["messages"]
        gold_entities = entities_by_pmid[doc_id]
        gold_text = text_by_pmid[doc_id]

        title_abs_gold_entities = sentence_wise(gold_text, gold_entities)

        extracted_entities, syn_err, val_err = postprocess_entity_outputs(
            extracted_entities_raw
        )
        predicted_entities[doc_id] = extracted_entities
        golden_entities[doc_id] = title_abs_gold_entities
        error_sents = set()
        correct_sents = set()
        for idx, _ in extracted_entities.items():
            if idx not in title_abs_gold_entities:
                wrong += 1
            if extracted_entities[idx] and idx in title_abs_gold_entities:

                if "diseases" in extracted_entities[idx]:
                    
                    # print(len(extracted_entities[idx]["diseases"]))
                    try:
            
                        num_extractions += len(extracted_entities[idx]["diseases"])
                        cur_correct = strict_match(
                            extracted_entities[idx]["diseases"],
                            title_abs_gold_entities[idx],
                        )

                        if cur_correct < len(extracted_entities[idx]["diseases"]):
                            error_sents.add(idx)

                        if cur_correct == len(extracted_entities[idx]["diseases"]):
                            correct_sents.add(idx)

                    except Exception as e:
                        pass
                    
                    overall_strict_correct += cur_correct

                else:
                    # if it has other predictions with diff format what do we do in evaluation?
                    pass

        all_syn_err += syn_err
        all_attr_err += attr_err
        all_val_err += val_err
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
    print(all_val_err)
    print(all_type_err)
    print(all_attr_err)
    True # 

    return overall_strict_prec, overall_strict_rec


def _is_entity_in_sentence(data_dict, offset):
    if data_dict["offsets"][0] >= offset[0] and data_dict["offsets"][1] <= offset[1]:
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
            sent_gold_entities.append(gold_entities[idx]["text"])
            idx += 1

        abs_gold_entities[str(i)] = sent_gold_entities
    return abs_gold_entities


def postprocess_entity_outputs(extracted_entities):
    postprocessed_entities = {}
    syn_err = 0
    val_err = 0
    for sent_num, extraction in extracted_entities.items():
        
        extraction = extraction.replace("\n", "")

        if "</json>" in extraction:
            extraction = extraction.split("</json>")[0]

        if "')" in extraction:
            extraction = extraction.split("')")[0]

        if "}']" in extraction:
            extraction = extraction.split("']")[0]

        if ", '<human>:" in extraction:
            extraction = extraction.split(", '<human>:")[0]

        if "\n<human>" in extraction:
            extraction = extraction.split("\n<human>")[0]

        if "<\n/human>" in extraction:
            extraction = extraction.split("<\n/human>")[0]
        
        if "</bot>" in extraction:
            extraction = extraction.split("</bot>")[0]
        
        if "</bot:>" in extraction:
            extraction = extraction.split("</bot:>")[0]
        
        if "<\n/bot>" in extraction:
            extraction = extraction.split("<\n/bot>")[0]

        if "</human>" in extraction:
            extraction = extraction.split("</human>")[0]

        if "<\n/human>" in extraction:
            extraction = extraction.split("<\n/human>")[0]

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

        try:
            extraction = ast.literal_eval(extraction)
            # Only reading the pedicted entities and returning them in the dict format and changing "disease" key to "diseases" for easier metric calculation.
            # if "disease" in extraction:
            #     extraction["diseases"] = extraction["disease"]
            #     del extraction["disease"]

            postprocessed_entities[sent_num] = extraction

        except SyntaxError as e:
            #print("This is a syntax error in postprocessing")
            print("--------")
            print(extraction)
            syn_err += 1
            #extraction = {"diseases": []}
            

        except ValueError as e:
            val_err += 1
            #extraction = {"diseases": []}

        #
        #postprocessed_entities[sent_num] = extraction

    return postprocessed_entities, syn_err, val_err


def strict_match(preds, golds):
    # do I need try and catch here?
    try: 
        preds = [x.lower() for x in preds]
        golds = [x.lower() for x in golds]
    except Exception as e:
        preds = []
        golds = []

    return len(set(preds).intersection(set(golds)))


def main(output_file):
    print(output_file)
    print("Loading dataset...")
    dataset = load_dataset("bigbio/ncbi_disease")["test"]
    extracted_entities = json.load(open(output_file))
    print(len(extracted_entities))
    entities_by_pmid = {}
    text_by_pmid = {}

    for idx, item in enumerate(dataset):
        pmid = item["pmid"]
        # title offset
        # text offset
        # correct this later?
        title = item["title"]
        abstract = item["abstract"]
        text = title + " " + abstract
        item["sent_offsets"] = []
        # Adding in abstract dict but this also includes title offsets
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            cur_offset = [start, end]
            item["sent_offsets"].append(cur_offset)

        text_by_pmid[pmid] = [text, item["sent_offsets"]]
        entities_by_pmid[pmid] = item["mentions"]

    prec, rec = eval_entity_extraction(extracted_entities, text_by_pmid, entities_by_pmid)

    return prec, rec


if __name__ == "__main__":
    main(output_file)
