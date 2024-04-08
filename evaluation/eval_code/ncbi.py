import json
from ipdb import set_trace
from datasets import load_dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer


def eval_entity_extraction(extractions, text_by_pmid, entities_by_pmid):
    # Iterate over all documents
    overall_strict_correct = 0.0
    predicted_entities = {}
    golden_entities = {}
    all_syn_err = 0
    wrong = 0
    all_key_err = 0
    all_val_err = 0
    all_type_err = 0
    all_attr_err = 0
    num_extractions = 0
    for doc_id in extractions:
        key_err = 0
        val_err = 0
        attr_err = 0
        type_err = 0
        extracted_entities_raw = extractions[doc_id]
        # here instead dict as a value, I have a list as a value, the first list is the entities and the second list is the entity types but I need this like a dic
        # or a tuple
        gold_entities = entities_by_pmid[doc_id]
        gold_text = text_by_pmid[doc_id]
        title_abs_gold_entities = sentence_wise(gold_text, gold_entities)
        extracted_entities, syn_err = postprocess_entity_outputs(extracted_entities_raw)
        predicted_entities[doc_id] = extracted_entities
        golden_entities[doc_id] = title_abs_gold_entities

        for idx, _ in extracted_entities.items():
            if idx not in title_abs_gold_entities:
                wrong += 1

            if extracted_entities[idx] and idx in title_abs_gold_entities:
                num_extractions += len(extracted_entities[idx]["diseases"])
                print(len(extracted_entities[idx]["diseases"]))

                cur_correct = strict_match(
                    extracted_entities[idx]["diseases"],
                    title_abs_gold_entities[idx],
                )

                overall_strict_correct += cur_correct

        all_syn_err += syn_err
        all_attr_err += attr_err
        all_val_err += val_err
        all_key_err += key_err
        all_type_err += type_err

    total_predictions = 0
    # idk how to do this

    for doc in predicted_entities:
        for sent in predicted_entities[doc]:
            length = len(predicted_entities[doc][sent])
            total_predictions += length

    overall_strict_prec = overall_strict_correct / total_predictions
    # fix recall
    total_num_extractions = 0
    for idx in entities_by_pmid:
        length = len(entities_by_pmid[idx])
        total_num_extractions += length

    overall_strict_rec = overall_strict_correct / total_num_extractions

    overall_strict_prec = round(overall_strict_prec, 4)
    overall_strict_rec = round(overall_strict_rec, 4)

    print(overall_strict_prec)
    print(overall_strict_rec)
    print(all_syn_err)
    print(all_key_err)
    print(all_val_err)
    print(all_type_err)
    print(all_attr_err)

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


def transform_list_to_dict(entity_list):
    transformed_dict = {"diseases": []}
    error = 0

    if entity_list:
        for i in range(len(entity_list)):
            if "disease" in entity_list[i]:
                transformed_dict["diseases"].append(entity_list[i]["disease"])
            elif "diseases" in entity_list[i]:
                transformed_dict["diseases"].append(entity_list[i]["diseases"])
            elif "text" in entity_list[i]:
                transformed_dict["diseases"].append(entity_list[i]["text"])

    else:
        print("Empty dict")

    return transformed_dict, error


def postprocess_entity_outputs(extracted_entities):
    postprocessed_entities = {}
    syn_err = 0
    for sent_num, extraction in extracted_entities.items():
        extraction = extracted_entities[sent_num]
        entity_list = []
        #extraction = "entity_list.append({'" + extraction

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

        try:
            exec(extraction)
        except:
            syn_err += 1
            pass

        processed_extractions, error = transform_list_to_dict(entity_list)
        postprocessed_entities[sent_num] = processed_extractions

    return postprocessed_entities, syn_err


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
    dataset = load_dataset("bigbio/ncbi_disease")["test"]
    extracted_entities = json.load(open(output_file))

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
    main()
