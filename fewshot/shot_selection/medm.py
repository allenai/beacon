import json
import specter
import random
from tqdm import tqdm
from ipdb import set_trace
from datasets import load_dataset, load_from_disk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from constants import DATA_DIR

def _is_entity_in_sentence(data_dict, offset):
    if (
        data_dict["offsets"][0][0] >= offset[0]
        and data_dict["offsets"][0][1] <= offset[1]
    ):
        return True
    else:
        return False


def sentence_wise(gold_text, gold_entities, id, split):
    # With the offsets from
    abs_gold_entities = {}
    offset_list = gold_text[1]
    gold_sents = gold_text[0]
    idx = 0

    for i, sent_offset in enumerate(offset_list):
        sent_gold_entities = []
        data_dict = {}
        while idx < len(gold_entities) and _is_entity_in_sentence(
            gold_entities[idx], sent_offset
        ):
            sent_gold_entities.append(gold_entities[idx])
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
        content = item["passages"]

        text = content[0]["text"][0] + " " + content[1]["text"][0]
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
    train_dataset = load_dataset("bigbio/medmentions")["test"]
    val_dataset = load_dataset("bigbio/medmentions", split="validation")
    all_train_dict = make_data_dict(train_dataset, split="train")
    val_dict = make_data_dict(val_dataset, split="val")

    seeds = [12, 23, 42]
    for seed in seeds:
        all_responses = {}
        random.seed(seed)
        res = dict(list(all_train_dict.items()))

        for doc_id, data in tqdm(val_dict.items()):
            for sent_id, sent_data in data.items():
                target_sent = sent_data["text"]
                top_sentences = specter.calculate_similarity(res, target_sent)
                all_responses[doc_id + "-" + sent_id] = top_sentences

        print("Saving JSON")
        with open(
            DATA_DIR / "fewshots/medm/val_k15_seed{seed}_spectre.json",
            "w",
        ) as f:
            json.dump(all_responses, f)

        with open(
            DATA_DIR / "fewshots/medm/test_full_spectre.json",
            "w",
        ) as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
