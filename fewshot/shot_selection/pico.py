import json
import nltk
import pandas as pd
import specter
from tqdm import tqdm
import random
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from constants import DATA_DIR


def _is_entity_in_sentence(data_dict, offset):
    if data_dict["start"] >= offset[0] and data_dict["end"] <= offset[1]:
        return True
    else:
        return False


def sentence_wise(gold_text, gold_entities, id, split):
    """
    This function works with gold extrations and looks at the sentence offsets saved in gold_text and buckets the entities from gold_entities into sentences.
    """
    abs_gold_entities = {}
    offset_list = gold_text[1]
    gold_sents = gold_text[0]

    idx = 0

    sorted_list = sorted(gold_entities, key=lambda x: x["start"])
    for i, sent_offset in enumerate(offset_list):
        sent_gold_entities = []
        data_dict = {}
        while idx < len(sorted_list) and _is_entity_in_sentence(
            sorted_list[idx], sent_offset
        ):
            sent_gold_entities.append(sorted_list[idx])
            idx += 1

        data_dict["text"] = gold_sents[i]
        data_dict["entities"] = sent_gold_entities
        if split == "train":
            abs_gold_entities[str(id) + "-" + str(i)] = data_dict
        if split == "val":
            abs_gold_entities[str(i)] = data_dict

    return abs_gold_entities


def make_data_dict(dataset, split):
    data_dict = {}
    for item in dataset:
        id = item["doc_id"]
        text = item["text"]

        item["sent_offsets"] = []
        all_sentences = []
        # Adding in abstract dict but this also includes title offsets
        for start, end in PunktSentenceTokenizer().span_tokenize(text):
            cur_offset = [start, end]
            sentence = text[start:end]
            item["sent_offsets"].append(cur_offset)
            all_sentences.append(sentence)

        gold_text = [all_sentences, item["sent_offsets"]]

        all_list = []
        gold_entities = []
        for each in item["entities"]:
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
                gold_entities.append(each_dict)

        sent_gold_dict = sentence_wise(gold_text, gold_entities, id, split=split)
        if split == "train":
            data_dict.update(sent_gold_dict)
        if split == "val":
            data_dict[id] = sent_gold_dict

    return data_dict


def main():
    train_dataset = load_dataset("bigbio/ebm_pico", split="test")
    train_dataset = load_from_disk(
        DATA_DIR / "subsamples/ebm_pico/test"
    )
    all_train_dict = make_data_dict(train_dataset, split="train")
    val_dict = make_data_dict(val_dataset, split="val")

    seeds = [12]
    for seed in seeds:
        all_responses = {}
        random.seed(seed)
        res = dict(random.sample(list(all_train_dict.items()), 15))
        res = dict(list(all_train_dict.items()))

        for doc_id, data in tqdm(val_dict.items()):
            for sent_id, sent_data in data.items():
                target_sent = sent_data["text"]
                top_sentences = specter.calculate_similarity(res, target_sent)
                all_responses[str(doc_id + "-" + sent_id)] = top_sentences

        print("Saving JSON")
        with open(
            DATA_DIR / "fewshots/pico/val_k15_seed{seed}spectre.json",
            "w",
        ) as f:
            json.dump(all_responses, f)

        print("saved json") 
        with open(
            DATA_DIR / "fewshots/pico/test_100_spectre.json",
            "w",
        ) as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
