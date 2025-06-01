import os
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def run_classification(data_folder, output_folder, labels_file, topics_file):
    """
    Classifies .txt files in `data_folder` and saves results to `output_folder`.
    Expects user to provide:
        - labels_file: path to a comma-separated .txt file
        - topics_file: path to a file with 'topic: description' per line
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read candidate labels
    with open(labels_file, 'r') as f:
        candidate_labels = [label.strip() for label in f.read().split(',')]

    # Read topic descriptions
    with open(topics_file, 'r') as f:
        lines = f.read().split('\n')
        topic_descriptions = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                topic_descriptions[key.strip()] = value.strip()

    # Zero-shot classifier
    device_id = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)

    # Semantic model
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda' if torch.cuda.is_available() else 'cpu')
    topic_embeddings = semantic_model.encode(list(topic_descriptions.values()), show_progress_bar=False)

    for fname in os.listdir(data_folder):
        if not fname.endswith('.txt'):
            continue

        with open(os.path.join(data_folder, fname), 'r', encoding='utf-8') as f:
            text = f.read()

        # Zero-shot (first 1000 chars to reduce cost)
        zero_result = classifier(text[:1000], candidate_labels)

        # Semantic similarity
        doc_embedding = semantic_model.encode([text])
        scores = cosine_similarity(doc_embedding, topic_embeddings)[0]
        semantic_sorted = sorted(zip(topic_descriptions.keys(), scores), key=lambda x: x[1], reverse=True)

        # Save
        with open(os.path.join(output_folder, f"result_{fname}"), 'w', encoding='utf-8') as out:
            out.write("Zero-Shot Classification:\n")
            for label, score in zip(zero_result['labels'][:5], zero_result['scores'][:5]):
                out.write(f"- {label}: {score:.3f}\n")

            out.write("\nSemantic Similarity:\n")
            for label, score in semantic_sorted[:5]:
                out.write(f"- {label}: {score:.3f}\n")

        print(f"Processed: {fname}")
