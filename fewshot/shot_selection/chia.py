import json
import nltk
import specter
import random
import pandas as pd
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_from_disk
from constants import DATA_DIR

def _is_entity_in_sentence(data_dict, offset):
    if (
        data_dict["offsets"][0][0] >= offset[0]
        and data_dict["offsets"][0][1] <= offset[1] + 1
    ):
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

    sorted_list = sorted(gold_entities, key=lambda x: x["offsets"])
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


def split_abstract_into_sentences(id, abstract):
    # Split the abstract into sentences using NLTK's sent_tokenize function
    sentences = filter(None, abstract.split("\n"))
    # Compute the sentence offsets
    offsets = []
    start = 0
    stuff = {}
    all_sentences = []
    for sentence in sentences:
        end = start + len(sentence)
        offsets.append((start, end))
        start = end + 2
        all_sentences.append(sentence)

    stuff[id] = all_sentences

    # error_file_path = OUTPUT_DIR / f"/error_analysis/chia/messedup_sents.txt"
    # with open(
    #     error_file_path,
    #     "a+",
    # ) as f:
    #     f.write(str(stuff))
    #     f.write("\n")

    return all_sentences, offsets


def make_data_dict(dataset, split):
    data_dict = {}
    for item in dataset:
        id = item["id"]
        text = item["text"]
        item["sent_offsets"] = []

        # Adding in abstract dict but this also includes title offsets
        sentences, offsets = split_abstract_into_sentences(id, text)

        item["sent_offsets"] = offsets

        gold_text = [sentences, item["sent_offsets"]]
        gold_entities = item["entities"]

        sent_gold_dict = sentence_wise(gold_text, gold_entities, id, split=split)

        if split == "train":
            data_dict.update(sent_gold_dict)
        if split == "val":
            data_dict[id] = sent_gold_dict

    return data_dict


def main():
    train_dataset = load_from_disk(
        ""replace with the chia data split path"/test.json"
    )
    val_dataset = load_from_disk(
        DATA_DIR / "subsamples/chia/val"
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
                all_responses[doc_id + "-" + sent_id] = top_sentences

        print("Saving JSON")
        with open(
            DATA_DIR / "fewshots/chia/val_k15_seed{seed}_spectre.json",
            "w",
        ) as f:
            json.dump(all_responses, f)
        
        print("saved json")
        with open(
            DATA_DIR / "fewshots/chia/test_full_spectre.json",
            "w",
        ) as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
