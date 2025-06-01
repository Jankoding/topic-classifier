import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor
import gc

def load_config_files(labels_file, topics_file):
    """Load candidate labels and topic descriptions from files"""
    # Load candidate labels
    with open(labels_file, 'r', encoding='utf-8') as f:
        candidate_labels = [line.strip() for line in f if line.strip()]
    
    # Load topic descriptions
    topic_descriptions = {}
    with open(topics_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                topic, description = line.strip().split(':', 1)
                topic_descriptions[topic.strip()] = description.strip()
    
    return candidate_labels, topic_descriptions

def batch_process_texts(texts, model_func, batch_size=8):
    """Process texts in batches to maximize GPU utilization"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = model_func(batch)
        results.extend(batch_results)
        
        # Clear GPU cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def prepare_text_chunks(text, tokenizer, max_chunk_tokens=850):
    """Prepare text chunks for processing"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        token_count = len(tokenizer.tokenize(sentence))
        
        if current_tokens + token_count > max_chunk_tokens:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = token_count
        else:
            current_chunk += " " + sentence
            current_tokens += token_count

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
    
def batch_zero_shot_classification(file_data_list, candidate_labels, device_id=0, batch_size=4):
    """Perform zero-shot classification on multiple files with GPU batching"""
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli", 
        device=device_id,
        batch_size=batch_size  # Enable batch processing
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    
    results = []
    
    # Prepare all chunks from all files
    all_chunks = []
    chunk_to_file_mapping = []
    
    for file_idx, (file_name, text) in enumerate(file_data_list):
        chunks = prepare_text_chunks(text, tokenizer)
        all_chunks.extend(chunks)
        chunk_to_file_mapping.extend([file_idx] * len(chunks))
        print(f"File {file_name}: {len(chunks)} chunks prepared")
    
    print(f"Processing {len(all_chunks)} total chunks in batches of {batch_size}...")
    
    # Process all chunks in batches
    chunk_results = []
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        try:
            # Process batch
            batch_outputs = classifier(batch_chunks, candidate_labels)
            chunk_results.extend(batch_outputs)
            print(f"Processed batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}")
            # Add dummy results for failed batch
            chunk_results.extend([{'labels': candidate_labels, 'scores': [0.0]*len(candidate_labels)} for _ in range(len(batch_chunks))])
    
    # Aggregate results by file
    file_chunk_counts = []
    chunk_idx = 0
    
    for file_idx, (file_name, text) in enumerate(file_data_list):
        file_chunks = prepare_text_chunks(text, tokenizer)
        num_chunks = len(file_chunks)
        file_chunk_counts.append(num_chunks)
        
        # Get chunk results for this file
        file_chunk_results = chunk_results[chunk_idx:chunk_idx + num_chunks]
        chunk_idx += num_chunks
        
        # Initialize score accumulators for each label
        label_scores = {label: [] for label in candidate_labels}
        
        # Collect scores for each label across all chunks
        for result in file_chunk_results:
            # Create mapping from label to score for this chunk
            chunk_label_scores = dict(zip(result['labels'], result['scores']))
            
            # For each candidate label, get its score (or 0 if not present)
            for label in candidate_labels:
                score = chunk_label_scores.get(label, 0.0)
                label_scores[label].append(score)
        
        # Calculate average scores for each label
        aggregated_scores = {
            label: sum(scores)/len(scores) if scores else 0.0
            for label, scores in label_scores.items()
        }
        
        # Sort labels by average score
        sorted_labels = sorted(
            aggregated_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results.append({
            'labels': [label for label, score in sorted_labels],
            'scores': [score for label, score in sorted_labels],
            'chunks_processed': len(file_chunk_results),
            'fallback_used': False
        })
    
    return results

def batch_semantic_similarity(file_data_list, topic_descriptions, device='cuda', batch_size=16):
    """Perform semantic similarity on multiple files with GPU batching"""
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Prepare topic data
    topics = list(topic_descriptions.keys())
    topic_texts = list(topic_descriptions.values())
    
    # Encode topic descriptions once
    print("Encoding topic descriptions...")
    topic_embeddings = model.encode(topic_texts, batch_size=batch_size, show_progress_bar=True)
    
    # Prepare all document texts
    all_texts = [text for _, text in file_data_list]
    
    # Encode all documents in batches
    print(f"Encoding {len(all_texts)} documents...")
    doc_embeddings = model.encode(all_texts, batch_size=batch_size, show_progress_bar=True)
    
    # Calculate similarities for all documents at once
    print("Calculating similarities...")
    all_similarities = cosine_similarity(doc_embeddings, topic_embeddings)
    
    # Process results
    results = []
    for i, similarities in enumerate(all_similarities):
        topic_scores = list(zip(topics, similarities))
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        
        results.append({
            'labels': [topic for topic, _ in topic_scores],
            'scores': [score for _, score in topic_scores]
        })
    
    return results

def write_zero_shot_results(file_name, text, result, output_path):
    """Write zero-shot classification results in the specified format"""
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # File metadata
        f_out.write(f"File: {file_name}\n")
        f_out.write(f"Text length: {len(text)} characters\n")
        
        # Chunking info if available
        if 'chunks_processed' in result:
            f_out.write(f"Chunks processed: {result['chunks_processed']}\n")
            if result.get('fallback_used', False):
                f_out.write("⚠️  Fallback processing used due to chunk processing errors\n")
        
        # Primary topic with confidence assessment
        primary_score = result['scores'][0]
        if primary_score >= 0.7:
            confidence_level = "HIGH"
        elif primary_score >= 0.4:
            confidence_level = "MODERATE"
        elif primary_score >= 0.2:
            confidence_level = "LOW"
        else:
            confidence_level = "VERY LOW"
        
        f_out.write(f"Primary topic: {result['labels'][0]} (confidence: {primary_score:.3f} - {confidence_level})\n")
        
        # Warning for low confidence
        if primary_score < 0.4:
            f_out.write(f"⚠️  WARNING: Low confidence classification. Document may not fit well into predefined categories.\n")
        
        # Top 5 topics with relevance indicators
        f_out.write("\nTop 5 topics:\n")
        for label, score in zip(result['labels'][:5], result['scores'][:5]):
            if score >= 0.4:
                confidence_indicator = "✓ RELEVANT"
            elif score >= 0.2:
                confidence_indicator = "~ MAYBE"
            else:
                confidence_indicator = "✗ UNLIKELY"
            f_out.write(f"  - {label}: {score:.3f} ({confidence_indicator})\n")
        
        # Confident predictions summary
        confident_topics = [(label, score) for label, score in zip(result['labels'], result['scores']) if score >= 0.4]
        if len(confident_topics) > 1:
            f_out.write(f"\nConfident predictions (≥0.4): {', '.join([label for label, _ in confident_topics])}\n")
        elif len(confident_topics) == 0:
            f_out.write(f"\nNo confident predictions found. This document may need manual categorization.\n")

def write_semantic_results(file_name, result, output_path):
    """Write semantic similarity results in the specified format"""
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(f"File: {file_name}\n")
        
        # Primary topic with similarity score
        primary_score = result['scores'][0]
        f_out.write(f"Primary topic: {result['labels'][0]} (similarity: {primary_score:.3f})\n")
        
        # Top 5 topics
        f_out.write("Top 5 topics:\n")
        for label, score in zip(result['labels'][:5], result['scores'][:5]):
            f_out.write(f"  - {label}: {score:.3f}\n")

def run_classification(data_folder, output_folder, labels_file, topics_file, 
                      zero_shot_batch_size=4, semantic_batch_size=16):
    """
    Run GPU-optimized batch topic classification
    
    Args:
        data_folder: Path to folder containing .txt files
        output_folder: Path to output folder for results
        labels_file: Path to file containing candidate labels (one per line)
        topics_file: Path to file containing topic descriptions (format: topic: description)
        zero_shot_batch_size: Batch size for zero-shot classification (smaller due to memory)
        semantic_batch_size: Batch size for semantic similarity (can be larger)
    """
    
    # Setup
    os.makedirs(output_folder, exist_ok=True)
    nltk.download('punkt', quiet=True)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_id = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {device}")
    
    # Load configuration
    candidate_labels, topic_descriptions = load_config_files(labels_file, topics_file)
    print(f"Loaded {len(candidate_labels)} candidate labels and {len(topic_descriptions)} topic descriptions")
    
    # Load all files
    file_names = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
    print(f"Found {len(file_names)} text files to process")
    
    # Read all files
    file_data_list = []
    for file_name in file_names:
        with open(os.path.join(data_folder, file_name), 'r', encoding='utf-8') as f:
            text = f.read()
        file_data_list.append((file_name, text))
    
    print("Starting batch processing...")
    
    # Run zero-shot classification in batches
    print("\n1. Running zero-shot classification...")
    zero_shot_results = batch_zero_shot_classification(
        file_data_list, candidate_labels, device_id, zero_shot_batch_size
    )
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run semantic similarity in batches
    print("\n2. Running semantic similarity...")
    semantic_results = batch_semantic_similarity(
        file_data_list, topic_descriptions, device, semantic_batch_size
    )
    
    # Write results in separate files for each method
    print("\n3. Writing results...")
    for i, (file_name, text) in enumerate(file_data_list):
        # Write zero-shot results
        zero_shot_path = os.path.join(output_folder, f"zero_shot_{file_name}")
        write_zero_shot_results(file_name, text, zero_shot_results[i], zero_shot_path)
        
        # Write semantic results
        semantic_path = os.path.join(output_folder, f"semantic_{file_name}")
        write_semantic_results(file_name, semantic_results[i], semantic_path)
    
    # Create summary
    summary_path = os.path.join(output_folder, "classification_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write("GPU BATCH TOPIC CLASSIFICATION SUMMARY\n")
        summary_file.write("="*60 + "\n\n")
        summary_file.write(f"Processed {len(file_names)} files using {device.upper()}\n")
        summary_file.write(f"Zero-shot batch size: {zero_shot_batch_size}\n")
        summary_file.write(f"Semantic batch size: {semantic_batch_size}\n\n")
        
        for i, (file_name, _) in enumerate(file_data_list):
            summary_file.write(f"File: {file_name}\n")
            summary_file.write("-" * (len(file_name) + 6) + "\n")
            
            # Zero-shot primary topic
            zs_primary = zero_shot_results[i]['labels'][0]
            zs_score = zero_shot_results[i]['scores'][0]
            summary_file.write(f"Zero-shot: {zs_primary} ({zs_score:.3f})\n")
            
            # Semantic primary topic
            sem_primary = semantic_results[i]['labels'][0]
            sem_score = semantic_results[i]['scores'][0]
            summary_file.write(f"Semantic: {sem_primary} ({sem_score:.3f})\n\n")
    
    print(f"\n✅ Processing complete!")
    print(f"Results saved in: {output_folder}")
    print("Files generated:")
    print("- zero_shot_*.txt: Zero-shot classification results")
    print("- semantic_*.txt: Semantic similarity results")
    print("- classification_summary.txt: Batch processing summary")

if __name__ == "__main__":
    # Example usage
    run_classification(
        data_folder="./data",
        output_folder="./results",
        labels_file="./config/labels.txt",
        topics_file="./config/topics.txt",
        zero_shot_batch_size=4,
        semantic_batch_size=16
    )
