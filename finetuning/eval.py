from ipdb import set_trace
import json
from constants import OUTPUT_DIR

path = OUTPUT_DIR /"/ncbi_seed12_method2/generated_predictions.json"

total_cur = 0 
total_num_extracted = 0 
total_num_gold = 0 

with open(path) as f:
    all_sentences = json.load(f)

for sentence in all_sentences:
    gold = sentence["gold"]
    extracted = sentence["output"]

    if "</s>" in extracted:
        extracted = extracted.split("</s")[0]
    extracted = extracted.replace('<pad>', '')

    extracted = extracted.strip()
    extracted = extracted.split(" <sep> ")

    if not gold is None:
        gold = gold.split(" <sep> ")

    if gold is None:
        gold = []
    if extracted is None:
        extracted = []

    total_num_gold += len(gold)
    total_num_extracted += len(extracted)

    print(extracted)
    print(gold)
    print("--------")

    try:
        total_cur += len(set(extracted).intersection(set(gold)))

    except:
        pass

print(total_num_extracted)
print(total_num_gold)
print(total_cur)

precision = total_cur / total_num_extracted
recall = total_cur / total_num_gold

f1_score = (2 * precision * recall) / (precision + recall)

print("prec:", precision)
print("rec:", recall)
print("f1 score:", f1_score)


