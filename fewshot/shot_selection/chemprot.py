import json
import nltk
import specter
import pandas as pd
from tqdm import tqdm
import random
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from constants import DATA_DIR

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


def sentence_wise(gold_text, gold_entities, id, split):
    """
    This function works with gold extrations and looks at the sentence offsets saved in gold_text and buckets the entities from gold_entities into sentences.
    """
    abs_gold_entities = {}
    offset_list = gold_text[1]
    gold_sents = gold_text[0]
    idx = 0

    gold_entities_list = convert_dict_to_list_of_dicts(gold_entities)
    sorted_list = sorted(gold_entities_list, key=lambda x: x["offsets"])
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
        id = item["pmid"]
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
        gold_entities = item["entities"]

        sent_gold_dict = sentence_wise(gold_text, gold_entities, id, split=split)
        if split == "train":
            data_dict.update(sent_gold_dict)
        if split == "val":
            data_dict[id] = sent_gold_dict

    return data_dict


def main():
    train_dataset = load_dataset("bigbio/chemprot")["test"]
    val_dataset = load_dataset("bigbio/chemprot")["validation"]
    all_train_dict = make_data_dict(train_dataset, split="train")
    val_dict = make_data_dict(val_dataset, split="val")

    seeds = [12]
    for seed in seeds:
        all_responses = {}
        # sample 100 from here
        random.seed(seed)
        res = dict(list(all_train_dict.items()))

        for doc_id, data in tqdm(val_dict.items()):
            for sent_id, sent_data in data.items():
                target_sent = sent_data["text"]
                top_sentences = specter.calculate_similarity(res, target_sent)
                all_responses[str(doc_id + "-" + sent_id)] = top_sentences
                # if they happen val sentence store

        print("Saving JSON")
        # problem with spaces
        
        with open(
            DATA_DIR / "fewshots/chemprot/val_k15_seed{seed}_spectre.json",
            "w",
        ) as f:
            json.dump(all_responses, f)
        
        with open(
            DATA_DIR / "fewshots/chemprot/test_full_spectre.json",
            "w",
        ) as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
