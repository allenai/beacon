import pickle
import torch
import numpy as np
from ipdb import set_trace
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoAdapterModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/specter2")
model = AutoAdapterModel.from_pretrained("allenai/specter2")

adapter_name = model.load_adapter(
    "allenai/specter2_proximity", source="hf", set_active=True
)


def calculate_similarity(train_data, sentence):
    # Create lists to store the train and test sentences

    train_sentences = []
    train_ids = []
    # Iterate over train_data to extract train sentences
    for doc_id, doc_data in train_data.items():
        train_sentences.append(doc_data["text"])
        train_ids.append(doc_id)

    train_inputs = tokenizer(
        train_sentences, padding=True, truncation=True, return_tensors="pt"
    )

    test_input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    # Get the embeddings

    with torch.no_grad():
        train_embeddings = model(
            **train_inputs, output_hidden_states=True, return_dict=True
        ).pooler_output

    with torch.no_grad():
        test_embedding = model(
            **test_input, output_hidden_states=True, return_dict=True
        ).pooler_output

    # Calculate cosine similarity between test sentence and train sentences
    cosine_similarities = cosine_similarity(test_embedding, train_embeddings).flatten()

    # Sort the similarities in descending order and get the top 5 indices
    top_indices = np.argsort(cosine_similarities)[::-1]

    # Retrieve the top 10 sentences and their respective similarity scores

    top_sentences = []
    for index in top_indices:
        string = train_ids[index]
        sent_data = train_data[string]
        score = cosine_similarities[index]
        top_sentences.append(
            {
                "doc_id+sent_id": string,
                "text": sent_data["text"],
                "entities": sent_data["entities"],
                "score": score.item(),
            }
        )

    return top_sentences
