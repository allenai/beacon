from ipdb import set_trace
import ast
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from itertools import islice
import json
from constants import DATA_DIR

nltk.download("punkt")  # Download necessary data for tokenization


ENTITY_MAP = {
    "ebm_pico": {"Participant": "PAR", "Intervention": "INT", "Outcome": "OUT"},
    "ncbi_disease": {"disease": "DIS"},
    "chemprot": {"CHEMICAL": "CHE", "GENE-N": "PRO", "GENE-Y": "PRO"},
    "bc5cdr": {"Chemical": "CHE", "Disease": "DIS"},
    "medmentions": {"disease": "DIS"},
}

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

def convert_offsets_to_word_level(text, start_offsets, end_offsets):
    words = nltk.word_tokenize(text)
    word_start_offsets = []
    word_end_offsets = []

    current_char = 0
    for word in words:
        word_start_offsets.append(current_char)
        current_char += len(word)
        word_end_offsets.append(current_char)

    # Map the character-level offsets to word-level offsets
    """
    word_start_indices = []
    word_end_indices = []
    for start, end in zip(start_offsets, end_offsets):
        for i, (word_start, word_end) in enumerate(
            zip(word_start_offsets, word_end_offsets)
        ):
            if start == word_start <= end:
                word_start_indices.append(i)
            if start < word_end <= end and word_start > start:
                word_end_indices.append(i)
    
    return words, word_start_indices, word_end_indices, word_start_offsets"""

    return words, word_start_offsets, word_end_offsets, start_offsets, end_offsets


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


def split_abstract_into_sentences(abstract):
    # Split the abstract into sentences using NLTK's sent_tokenize function
    sentences = filter(None, abstract.split("\n"))

    # Compute the sentence offsets
    end_offsets = []
    start = 0
    for sentence in sentences:
        end = start + len(sentence)
        end_offsets.append(end)
        start = end + 2
    return end_offsets, sentences


def json_to_conll2003(json_data, dataset):
    conll_lines = []

    set_trace()
    for entry in json_data:
        if dataset == "medm":
            doc_id = entry["pmid"]
            content = entry["passages"]
            text = content[0]["text"][0] + " " + content[1]["text"][0]
            entities = entry["entities"]

        elif dataset == "ncbi":
            doc_id = entry["pmid"]
            text = entry["title"] + " " + entry["abstract"]
            entities = entry["mentions"]

        elif dataset == "cdr":
            doc_id = entry["passages"][0]["document_id"]
            if entry["passages"][0]["type"] == "title":
                title = entry["passages"][0]["text"]
            abstract = entry["passages"][1]["text"]
            text = title + " " + abstract
            entities = entry["passages"][0]["entities"] + entry["passages"][1]["entities"]

        elif dataset == "chemprot":
            doc_id = entry["pmid"]
            text = entry["text"]
            entities = entry["entities"]

        elif dataset == "chia":
            doc_id = entry["id"]
            text = entry["text"]
            entities = entry["entities"]

        counter = 0
        sentences = []
        sent_end_offsets = []

        if dataset == "chia":
            sent_end_offsets, sentences = split_abstract_into_sentences(text)
            set_trace()

        else:
            for start, end in PunktSentenceTokenizer().span_tokenize(text):
                sentence = text[start:end]
                sentences.append(sentence)
                sent_end_offsets.append(end)

        if dataset == "ncbi":
            (
                words,
                word_start_indices,
                word_end_indices,
                word_start_offsets,
            ) = convert_offsets_to_word_level(
                text,
                [entity["offsets"][0] for entity in entities],
                [entity["offsets"][1] for entity in entities],
            )

        if dataset == "cdr" or dataset == "medm":
            (
                words,
                word_start_offset,
                word_end_offset,
                start_offsets,
                end_offsets
            ) = convert_offsets_to_word_level(
                text,
                [entity["offsets"][0][0] for entity in entities],
                [entity["offsets"][0][1] for entity in entities],
            )

        if dataset == "chemprot":
            gold_entities_list = convert_dict_to_list_of_dicts(entities)
            entities = sorted(gold_entities_list, key=lambda x: x["offsets"])

            (
                words,
                word_start_indices,
                word_end_indices,
                word_start_offsets,
            ) = convert_offsets_to_word_level(
                text,
                [entity["offsets"][0] for entity in entities],
                [entity["offsets"][1] for entity in entities],
            )

        # Initialize a list to store entity labels for each word
        entity_labels = ["O"] * len(words)

        if dataset =="cdr":

            for start_idx, end_idx in zip(
                    word_start_offset,
                    word_end_offset,
            ):

                # word start => entity_start; then its in the entity but also word_ end <= entity_end
                # word start = entity start for B tag
                entity_labels[start_idx] = f"B-{ENTITY_MAP[dataset][entity_type]}"
                for i in word_end_indices:
                    entity_labels[i] = f"I-{ENTITY_MAP[dataset][entity_type]}"



        if dataset == "ncbi_disease" or dataset == "medmentions":
            entity_type = "disease"

            for start_idx, end_idx in zip(
                word_start_indices,
                word_end_indices,
            ):
                entity_labels[start_idx] = f"B-{ENTITY_MAP[dataset][entity_type]}"
                for i in word_end_indices:
                    entity_labels[i] = f"I-{ENTITY_MAP[dataset][entity_type]}"

        else:
            for entity_type, start_idx, end_idx in zip(
                [entity["type"] for entity in entities],
                word_start_indices,
                word_end_indices,
            ):
                # Set B- and I- labels for entitie
                entity_labels[start_idx] = f"B-{ENTITY_MAP[dataset][entity_type]}"
                for i in word_end_indices:
                    entity_labels[i] = f"I-{ENTITY_MAP[dataset][entity_type]}"

        # Create CoNLL lines
        conll_lines.append("\n")
        conll_lines.append(f"-DOCSTART- ({doc_id})\n")

        for i, (word, label) in enumerate(zip(words, entity_labels)):
            if word_start_offsets[i] > sent_end_offsets[counter]:
                conll_lines.append("\n")
                counter += 1
            conll_lines.append(f"{word} NN O {label}")

        # pico NO bio tagging
        # Add an empty line to separate sentences

    # conll_lines.append("\n\n")

    return "\n".join(conll_lines)


def main():
    datasets = ["cdr", "chemprot", "ncbi", "medm", "pico", "chia"]
    splits = ["val", "test", "train"]
    seeds = [12, 23, 42]
    k = 5


    for dataset in datasets:
        for split in splits: 
            dataset = load_dataset("bigbio/bc5cdr")["test"]
            with open(dataset) as user_file:
                file_contents = user_file.read()

            data = [json.loads(s) for s in file_contents.split("\n") if s]


            conll_lines = json_to_conll2003(data, dataset)

            with open(
                DATA_DIR / f"/finetune/{dataset}/fewshot-ft/{split}_full.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(conll_lines)
            print(f"Saved {dataset} and {split}")


if __name__ == "__main__":
    main()