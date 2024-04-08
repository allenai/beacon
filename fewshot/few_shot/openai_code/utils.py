import json
import ast
from ipdb import set_trace
from collections import defaultdict
from contextlib import ContextDecorator
import time
from itertools import islice
import random
from constants import DATA_DIR

def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


class timeit(ContextDecorator):
    def __init__(self, var="", echo=False) -> None:
        super().__init__()
        self.var = var
        self.echo = echo

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *exc):
        if self.echo:
            print(f"Time taken for {self.var}: {time.time() - self.start_time}")
        return


class OutputFormatter(object):
    def __init__(self, dataset, seed) -> None:
        # topk_shots is for retieved shots
        # topk_path = DATA_DIR / "fewshots/{dataset}/val_k15_seed{seed}_spectre.json"
        path = DATA_DIR / "fewshots/{dataset}/val_15_seed{seed}_spectre.json"

        # with open(path) as user_file:
        #     file = user_file.read()
        #     self.all_train_examples = ast.literal_eval(file)

        with open(path) as user_file:
            file = user_file.read()
            self.top_k_examples = ast.literal_eval(file)

        self.dataset = dataset
        self.seed = seed

    def format_output(self, key_id, k):
        formatted_shots = {}
        # if k == 100:
        #     key_id = None
        #     top_shots = self.all_train_examples
        #     top_shots = [v | {"doc_id+sent_id": k} for k, v in top_shots.items()]

        # else:
        try:
            # l = list(self.top_k_examples.items())
            # random.shuffle(l)
            # d_shuffled = dict(l)
            top_shots = take(k, self.top_k_examples.items())
            # top_shots = self.top_k_examples[key_id][:k]

        except:
            
            pass

        if self.dataset == "pico":
            for i in range(len(top_shots)):
                each_shot = defaultdict(list)
                if isinstance(top_shots[i], list):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], tuple):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], dict):
                    item_id = top_shots[i]["doc_id+sent_id"]
                    item_stuff = top_shots[i]

                each_shot["id"] = item_id
                each_shot["text"] = item_stuff["text"]

                for j in range(len(item_stuff["entities"])):
                    if item_stuff["entities"][j]["annotation_type"] == "Population":
                        each_shot["population"].append(
                            item_stuff["entities"][j]["text"]
                        )
                    elif item_stuff["entities"][j]["annotation_type"] == "Intervention":
                        each_shot["intervention"].append(
                            item_stuff["entities"][j]["text"]
                        )
                    elif item_stuff["entities"][j]["annotation_type"] == "Comparator":
                        each_shot["comparator"].append(
                            item_stuff["entities"][j]["text"]
                        )
                    elif item_stuff["entities"][j]["annotation_type"] == "Outcome":
                        each_shot["outcome"].append(item_stuff["entities"][j]["text"])

                formatted_shots[i] = each_shot

        if self.dataset == "chia":
            formatted_shots = []
            for i in range(len(top_shots)):
                each_shot = defaultdict(list)
                if isinstance(top_shots[i], list):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], tuple):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], dict):
                    item_id = top_shots[i]["doc_id+sent_id"]
                    item_stuff = top_shots[i]

                each_shot["id"] = item_id
                each_shot["text"] = item_stuff["text"]

                each_shot = {
                    "id": item_id,
                    "text": item_stuff["text"],
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

                for entity in item_stuff["entities"]:
                    annotation_type = entity["type"]
                    each_shot[annotation_type.lower()].append(entity["text"][0])

                formatted_shots.append(each_shot)

        return formatted_shots
