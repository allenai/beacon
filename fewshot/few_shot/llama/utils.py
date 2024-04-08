import json
import pandas as pd
import ast
import random
from ipdb import set_trace
from collections import defaultdict
from contextlib import ContextDecorator
import time
from itertools import islice
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
        # topk_path = DATA_DIR / "fewshots/{dataset}/val_k15_seed{seed}_spectre.json"
        path = DATA_DIR / "fewshots/{dataset}/val_15_seed{seed}_spectre.json"

        with open(path) as user_file:
            file = user_file.read()
            self.random_k_examples = ast.literal_eval(file)

        # with open(topk_path, "r") as f:
        #     for line in f:
        #         
        #         self.top_k_examples = json.loads(line)

        # with open(topk_path) as user_file:
        #     file = user_file.read()
        #     self.top_k_examples = ast.literal_eval(file)

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
            top_shots = take(k, self.random_k_examples.items())
            # top_shots = self.random_k_examples[key_id][:k]

        except:
            
            pass

        if self.dataset == "cdr":
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
                    if item_stuff["entities"][j]["type"] == "Disease":
                        each_shot["diseases"].append(
                            item_stuff["entities"][j]["text"][0]
                        )
                    elif item_stuff["entities"][j]["type"] == "Chemical":
                        each_shot["chemicals"].append(
                            item_stuff["entities"][j]["text"][0]
                        )

                formatted_shots[i] = each_shot

        if self.dataset == "chemprot":
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
                    if item_stuff["entities"][j]["type"] == "GENE-N":
                        each_shot["proteins"].append(item_stuff["entities"][j]["text"])
                    elif item_stuff["entities"][j]["type"] == "GENE-Y":
                        each_shot["proteins"].append(item_stuff["entities"][j]["text"])
                    elif item_stuff["entities"][j]["type"] == "CHEMICAL":
                        each_shot["chemicals"].append(item_stuff["entities"][j]["text"])

                formatted_shots[i] = each_shot

        if self.dataset == "medm":
            for i in range(len(top_shots)):
                if isinstance(top_shots[i], list):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], tuple):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], dict):
                    item_id = top_shots[i]["doc_id+sent_id"]
                    item_stuff = top_shots[i]

                each_shot = defaultdict(list)
                each_shot["id"] = item_id
                each_shot["text"] = item_stuff["text"]

                for j in range(len(item_stuff["entities"])):
                    each_shot["entities"].append(item_stuff["entities"][j]["text"][0])
                
                formatted_shots[i] = each_shot

        if self.dataset == "ncbi":
            for i in range(len(top_shots)):
                if isinstance(top_shots[i], list):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], tuple):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], dict):
                    item_id = top_shots[i]["doc_id+sent_id"]
                    item_stuff = top_shots[i]

                each_shot = defaultdict(list)
                each_shot["id"] = item_id
                each_shot["text"] = item_stuff["text"]

                for j in range(len(item_stuff["entities"])):
                    each_shot["diseases"].append(item_stuff["entities"][j])
                
                formatted_shots[i] = each_shot

        if self.dataset == "pico":

            for i in range(len(top_shots)):
                if isinstance(top_shots[i], list):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], tuple):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], dict):
                    item_id = top_shots[i]["doc_id+sent_id"]
                    item_stuff = top_shots[i]

                each_shot = defaultdict(list)
                each_shot["id"] = item_id
                each_shot["text"] = item_stuff["text"]
    
                for j in range(len(item_stuff["entities"])):
                   
                    if item_stuff["entities"][j]["annotation_type"] == "Participant":
                        each_shot["population"].append(
                            item_stuff["entities"][j]["text"]
                        )
                    elif (
                        item_stuff["entities"][j]["annotation_type"] == "Intervention"
                    ):
                        each_shot["intervention"].append(
                            item_stuff["entities"][j]["text"]
                        )
                    elif item_stuff["entities"][j]["annotation_type"] == "Comparator":
                        each_shot["comparator"].append(
                            item_stuff["entities"][j]["text"]
                        )
                    elif item_stuff["entities"][j]["annotation_type"] == "Outcome":
                        each_shot["outcome"].append(
                            item_stuff["entities"][j]["text"]
                        )
     
                formatted_shots[i] = each_shot

        if self.dataset == "chia":

            for i in range(len(top_shots)):
                if isinstance(top_shots[i], list):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], tuple):
                    item_id = top_shots[i][0]
                    item_stuff = top_shots[i][1]

                elif isinstance(top_shots[i], dict):
                    item_id = top_shots[i]["doc_id+sent_id"]
                    item_stuff = top_shots[i]

                each_shot = defaultdict(list)
                each_shot["id"] = item_id
                each_shot["text"] = item_stuff["text"]

                for j in range(len(item_stuff["entities"])):
                    data_type = item_stuff["entities"][j]["type"].lower()
                    each_shot[data_type].append(item_stuff["entities"][j]["text"][0])

                formatted_shots[i] = each_shot

      
        return formatted_shots

