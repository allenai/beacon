import json
from calls import openai_call
from tqdm import tqdm
from datasets import load_from_disk
from prompts import chia
from constants import OUTPUT_DIR

def main():
    all_responses = {}
    prompt = chia.CODE_TEXT
    dataset = load_from_disk(
        ""replace with the chia data split path"/val.hf"
    )
    count = 0
    for item in tqdm(dataset):
        iid = item["id"]
        all_text = item["text"]
        sent_text = filter(None, all_text.split("\n"))
        # sent_text = nltk.sent_tokenize(text)
        sent_response = {}
        if item["id"][-3:] == "exc":
            criterion = """
                def named_entity_recognition(input_text):
                \""" Given the definitions of entities and following Exclution criterion for clinical trial, """
        else:
            criterion = """
                def named_entity_recognition(input_text):
                \""" Given the following Inclusion criterion for clinical trial, """
        for idx, text in enumerate(sent_text):
            user_input = (
                criterion
                + prompt
                + "input_test = "
                + text
                + "\nentity_list = [] \n # extracted entities \n entity_list.append({' "
            )
            response = openai_call.generate_text(user_input, temperature=0, max_tokens=128)
            sent_response[idx] = response

        all_responses[iid] = sent_response
        count += 1

        if (count % 100 == 0) or (count == len(dataset)):
            print(f"Num of test datapoints: {count}")
            with open(
                OUTPUT_DIR / f"/final_zs/chia/gpt/zs_text_code_{count}.json",
                "w",
            ) as f:
                json.dump(all_responses, f)


if __name__ == "__main__":
    main()
