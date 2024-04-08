import utils
import pandas as pd
from ipdb import set_trace
from constants import DATA_DIR


PROMPTS = {
"cdr" : """Given the sentence from an abstract, extract all the chemicals and diseases. Generate the output in this format: ['entity1:entity_type1',  'entity2:entity_type2'] """, 
"chemprot" : """Given the sentence from an abstract, extract all the chemicals and proteins. Generate the output in this format: ['entity1:entity_type1',  'entity2:entity_type2'] """, 
"ncbi" : """Given the sentence from an abstract, extract all the diseases. \n Generate the output in this format: entity1 <sep> entity2.""",
"medm" : """Given the sentence from an abstract, extract all the entities from the abstract which can be found in a UMLS. Generate the output in this format: ['entity1', 'entity2'] """, 
"pico" : """Given the sentence from an abstract, extract all the extract the Population, Intervention, Comparator and Outcome spans. Generate the output in this format: ['entity1:entity_type1',  'entity2:entity_type2', 'entity3:entity_type3', 'entity4:entity_type4'] """, 
"chia" : """Given the sentence from an abstract, extract the following entity types if present: Condition, Device, Drug, Measurement, Mood, Multiplier, Negation, Observation, Person, Procedure, Qualifier, Reference_point, Scope, Temporal, Value, Visit (in alphabetical order). Generate the output in this format: ['entity1:entity_type1',  'entity2:entity_type2'] """
}

def main(dataset, k, seed, split):

    
    output_formatter = utils.OutputFormatter(dataset=dataset, seed=seed, split=split)
    formatted_shots = output_formatter.format_output(k=k)
    # make a dataframe 
    df = pd.DataFrame(columns=['Input', 'Output'])
    for i in range(len(formatted_shots)):
        text = formatted_shots[i]["text"]
        inp = PROMPTS[f"{dataset}"] + "\nSentence: " + text + "\n\nExtracted entities: "
        # output = str(formatted_shots[i]["entities"])
        output = ' <sep> '.join(formatted_shots[i]["entities"])
        df2_dict = {'Input': inp, 'Output': output}
        df2 = pd.DataFrame(df2_dict, index=[0])
        df = pd.concat([df, df2], ignore_index = True)
        # append input and output to the dataframe

    path = DATA_DIR / f"/ft_data/{dataset}/ft_method2_{k}shot_seed{seed}_{split}.csv"
    df.to_csv(path, index=False)
    print(f"saved to {path}")

def main_test(dataset, split):
    
    output_formatter = utils.OutputFormatter(dataset=dataset, split=split)
    formatted_shots = output_formatter.format_output()
    # make a dataframe 
    df = pd.DataFrame(columns=['Input', 'Output'])
    for i in range(len(formatted_shots)):
        text = formatted_shots[i]["text"]
        inp = PROMPTS[f"{dataset}"] + " Sentence: " + text + " Extracted entities: "
        #output = str(formatted_shots[i]["entities"])
        output = ' <sep> '.join(formatted_shots[i]["entities"])
        df2_dict = {'Input': inp, 'Output': output}
        df2 = pd.DataFrame(df2_dict, index=[0])
        df = pd.concat([df, df2], ignore_index = True)
        # append input and output to the dataframe


    path = DATA_DIR / f"/ft_data/{dataset}/ft_full_method2_{split}.csv"
    df.to_csv(path, index=False)
    print(f"saved to {path}")


if __name__ == "__main__":

    main_test(dataset="ncbi", split="test")

