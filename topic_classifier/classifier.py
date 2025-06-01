import os
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')

def run_classification(data_folder, output_folder, labels_file=None, topics_file=None):
    os.makedirs(output_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_id = 0 if torch.cuda.is_available() else -1

    print(f"Using device: {device}")

    # Load labels
    if labels_file:
        with open(labels_file, 'r') as f:
            candidate_labels = [label.strip() for label in f.read().split(',')]
    else:
        candidate_labels = [
            "trump", "russia", "war", "peace", "trade", 
            "finance", "terrorism", "elections"
        ]

    # Load topic descriptions
    if topics_file:
        with open(topics_file, 'r') as f:
            lines = f.read().split('\n')
            topic_descriptions = {
                key.strip(): value.strip()
                for line in lines if ':' in line
                for key, value in [line.split(':', 1)]
            }
    else:
        topic_descriptions = {
            "technology": "programming, software, computers, coding, artificial intelligence, machine learning",
            "business": "company, marketing, sales, finance, strategy, management, economics",
            "health": "medical, healthcare, fitness, nutrition, wellness, disease, treatment",
            "education": "learning, teaching, school, university, course, training, academic",
            "entertainment": "movies, music, games, fun, leisure, celebrities, shows",
            "sports": "football, basketball, athletics, competition, exercise, team, player",
            "politics": "government, election, policy, law, democracy, political, voting",
            "science": "research, experiment, discovery, scientific, biology, physics, chemistry",
            "travel": "vacation, trip, tourism, destination, adventure, journey, explore",
            "food": "cooking, recipe, restaurant, cuisine, meal, ingredient, dining",
            "personal": "diary, thoughts, feelings, life, experience, reflection, personal story",
            "tutorial": "how to, guide, instructions, step by step, learn, tutorial, walkthrough",
            "news": "breaking news, current events, journalism, reporting, headlines, updates"
        }

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_id)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    topic_embeddings = semantic_model.encode(list(topic_descriptions.values()), show_progress_bar=False)
    topic_keys = list(topic_descriptions.keys())

    file_names = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
    print(f"Processing {len(file_names)} files...")

    for file_name in file_names:
        print(f"\n📄 File: {file_name}")
        with open(os.path.join(data_folder, file_name), 'r', encoding='utf-8') as f:
            text = f.read()

        ### --- Zero-Shot Classification with Chunking ---
        sentences = sent_tokenize(text)
        chunks, current_chunk, token_count = [], "", 0

        for sent in sentences:
            sent_tokens = tokenizer.tokenize(sent)
            if token_count + len(sent_tokens) > 850:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent
                token_count = len(sent_tokens)
            else:
                current_chunk += " " + sent
                token_count += len(sent_tokens)
        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"→ Chunked into {len(chunks)} sections")

        chunk_results = []
        for i, chunk in enumerate(chunks):
            try:
                print(f"→ Classifying chunk {i+1}/{len(chunks)}")
                chunk_result = classifier(chunk, candidate_labels)
                chunk_results.append(chunk_result)
            except Exception as e:
                print(f"⚠️ Error in chunk {i}: {e}")

        # Aggregate zero-shot results
        label_scores = {}
        for res in chunk_results:
            for label, score in zip(res['labels'], res['scores']):
                label_scores.setdefault(label, []).append(score)

        aggregated = {
            label: np.mean(scores) for label, scores in label_scores.items()
        }
        zero_sorted = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        zero_labels, zero_scores = zip(*zero_sorted) if zero_sorted else ([], [])

        ### --- Semantic Similarity ---
        doc_embedding = semantic_model.encode([text])
        similarity_scores = cosine_similarity(doc_embedding, topic_embeddings)[0]
        sem_sorted = sorted(zip(topic_keys, similarity_scores), key=lambda x: x[1], reverse=True)

        ### --- Write output ---
        out_path = os.path.join(output_folder, f"classification_{file_name}")
        with open(out_path, 'w', encoding='utf-8') as out:
            out.write(f"File: {file_name}\n")
            out.write(f"Length: {len(text)} characters\n")
            out.write(f"Chunks processed: {len(chunk_results)}\n\n")

            # Zero-shot results
            out.write("=== Zero-shot Classification ===\n")
            if zero_labels:
                top_label = zero_labels[0]
                top_score = zero_scores[0]
                conf = "HIGH" if top_score >= 0.7 else "MODERATE" if top_score >= 0.4 else "LOW"
                out.write(f"Primary topic: {top_label} ({top_score:.3f} - {conf})\n")
                if top_score < 0.4:
                    out.write("⚠️  WARNING: Low confidence classification.\n")
                out.write("\nTop 5 Topics:\n")
                for label, score in zip(zero_labels[:5], zero_scores[:5]):
                    symbol = "✓" if score >= 0.4 else "~" if score >= 0.2 else "✗"
                    out.write(f"  {symbol} {label}: {score:.3f}\n")
            else:
                out.write("Zero-shot classification failed.\n")

            # Semantic similarity
            out.write("\n=== Semantic Similarity ===\n")
            out.write(f"Primary topic: {sem_sorted[0][0]} (similarity: {sem_sorted[0][1]:.3f})\n")
            out.write("\nTop 5 Topics:\n")
            for label, score in sem_sorted[:5]:
                out.write(f"  - {label}: {score:.3f}\n")

    ### --- Create Summary ---
    summary_path = os.path.join(output_folder, "classification_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as summary:
        summary.write("TOPIC CLASSIFICATION SUMMARY\n")
        summary.write("=" * 40 + "\n\n")
        for file_name in file_names:
            result_path = os.path.join(output_folder, f"classification_{file_name}")
            summary.write(f"File: {file_name}\n")
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Primary topic:") or line.startswith("=== Semantic Similarity ==="):
                            summary.write(line)
                        if "Primary topic:" in line and "Semantic" not in line:
                            break
            except Exception as e:
                summary.write(f"Could not read file: {e}\n")
            summary.write("\n")
