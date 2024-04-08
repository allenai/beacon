import random
import pandas as pd
import ast
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
    def __init__(self, dataset, seed, split) -> None:

        path = DATA_DIR / f"fewshots/{dataset}/{split}_15_seed{seed}_spectre.json"  
        print(path)     
        with open(path) as user_file:
            file = user_file.read()
            self.random_k_examples = ast.literal_eval(file)

        self.dataset = dataset
        self.seed = seed

    def format_output(self, k):
        formatted_shots = {}

        try:
            top_shots = take(k, self.random_k_examples.items())

        except:
            set_trace()
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
                
                # ['seizures:disease', 'cocaine:chemical']

                for j in range(len(item_stuff["entities"])):
                    if item_stuff["entities"][j]["type"] == "Disease":
                        each_shot["entities"].append(item_stuff["entities"][j]["text"][0] + ":disease")
                    elif item_stuff["entities"][j]["type"] == "Chemical":
                        each_shot["entities"].append(item_stuff["entities"][j]["text"][0] +  ":chemical")

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
                    if item_stuff["entities"][j]["type"] == "CHEMICAL":
                        each_shot["entities"].append(item_stuff["entities"][j]["text"] + ":chemical")
                    else:
                        each_shot["entities"].append(item_stuff["entities"][j]["text"] +  ":protein")

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
                    each_shot["entities"].append(item_stuff["entities"][j])

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
                        each_shot["entities"].append(item_stuff["entities"][j]["text"] + ":population")

                    elif item_stuff["entities"][j]["annotation_type"] == "Intervention":
                        each_shot["entities"].append(item_stuff["entities"][j]["text"] + ":intervention")

                    elif item_stuff["entities"][j]["annotation_type"] == "Comparator":
                        each_shot["entities"].append(item_stuff["entities"][j]["text"] + ":comparator")

                    elif item_stuff["entities"][j]["annotation_type"] == "Outcome":
                        each_shot["entities"].append(item_stuff["entities"][j]["text"] + ":outcome")

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
                    each_shot["entities"].append(item_stuff["entities"][j]["text"][0] + f":{data_type}")

                formatted_shots[i] = each_shot


        return formatted_shots
