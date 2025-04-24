import csv
import random
from nltk.tag import hmm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
from itertools import combinations

# Download required NLTK data
nltk.download('punkt')

def load_and_preprocess_data(file_path: str) -> List[List[Tuple[str, str]]]:
    """
    Load and preprocess the data from CSV file.
    Args:
        file_path: Path to the CSV file
    Returns:
        List of sequences, where each sequence is a list of (word, tag) tuples
    """
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            if not row:  # Skip empty rows
                continue
                
            # Get non-empty columns
            selected_columns = [col for col in row[1:] if col.strip()]
            
            # Process each column to create word-tag pairs
            curr = []
            for s in selected_columns:
                try:
                    parts = s.split(",")
                    if len(parts) != 2:
                        continue
                        
                    word = parts[0].strip().lower()
                    tag = parts[1].strip()
                    
                    if word and tag:  # Only add if both word and tag are non-empty
                        curr.append((word, tag))
                except Exception as e:
                    print(f"Warning: Error processing column '{s}': {str(e)}")
                    continue
            
            if curr:  # Only add sequences that have at least one valid word-tag pair
                data.append(curr)
    
    return data

def evaluate_bleu(reference: List[List[str]], candidate: List[List[str]]) -> float:
    """
    Calculate BLEU score between reference and candidate sequences.
    Args:
        reference: List of reference sequences
        candidate: List of candidate sequences
    Returns:
        BLEU score
    """
    smoothie = SmoothingFunction().method1
    scores = []
    
    for ref, cand in zip(reference, candidate):
        # Convert sequences to strings and tokenize
        ref_str = ' '.join(ref)
        cand_str = ' '.join(cand)
        
        ref_tokens = nltk.word_tokenize(ref_str)
        cand_tokens = nltk.word_tokenize(cand_str)
        
        # Calculate BLEU score for this pair
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
        scores.append(score)
    
    # Return average BLEU score
    return sum(scores) / len(scores) if scores else 0.0

def evaluate_rouge(reference: List[List[str]], candidate: List[List[str]]) -> dict:
    """
    Calculate ROUGE scores between reference and candidate sequences.
    Args:
        reference: List of reference sequences
        candidate: List of candidate sequences
    Returns:
        Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    # Convert sequences to sentences
    reference_sentences = [' '.join(seq) for seq in reference]
    candidate_sentences = [' '.join(seq) for seq in candidate]
    
    # Calculate ROUGE scores
    rouge = Rouge()
    scores = rouge.get_scores(candidate_sentences, reference_sentences, avg=True)
    return scores

def evaluate_sequence_metrics(reference: List[List[str]], candidate: List[List[str]]) -> Dict:
    """
    Calculate sequence-level metrics.
    Args:
        reference: List of reference sequences
        candidate: List of candidate sequences
    Returns:
        Dictionary containing sequence-level metrics
    """
    exact_matches = 0
    partial_matches = 0
    total_sequences = len(reference)
    
    for ref, cand in zip(reference, candidate):
        # Exact match
        if ref == cand:
            exact_matches += 1
        
        # Partial match (at least 50% of tags correct)
        correct_tags = sum(1 for r, c in zip(ref, cand) if r == c)
        if correct_tags / len(ref) >= 0.5:
            partial_matches += 1
    
    return {
        'exact_match': exact_matches / total_sequences if total_sequences > 0 else 0,
        'partial_match': partial_matches / total_sequences if total_sequences > 0 else 0
    }

def analyze_errors(true_tags: List[str], pred_tags: List[str]) -> Dict:
    """
    Analyze error patterns in predictions.
    Args:
        true_tags: List of true tags
        pred_tags: List of predicted tags
    Returns:
        Dictionary containing error analysis
    """
    error_patterns = defaultdict(int)
    confusion_pairs = defaultdict(int)
    
    for true, pred in zip(true_tags, pred_tags):
        if true != pred:
            error_patterns[(true, pred)] += 1
            confusion_pairs[f"{true}->{pred}"] += 1
    
    return {
        'error_patterns': dict(error_patterns),
        'confusion_pairs': dict(confusion_pairs)
    }

def plot_error_analysis(error_patterns: Dict, confusion_pairs: Dict):
    """Plot error analysis visualizations."""
    # Plot most common error patterns
    plt.figure(figsize=(12, 6))
    patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    plt.bar([str(p) for p, _ in patterns], [c for _, c in patterns])
    plt.title('Top 10 Error Patterns')
    plt.xlabel('(True Tag, Predicted Tag)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('error_patterns.png')
    plt.close()
    
    # Plot confusion pairs
    plt.figure(figsize=(12, 6))
    pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    plt.bar([p for p, _ in pairs], [c for _, c in pairs])
    plt.title('Top 10 Confusion Pairs')
    plt.xlabel('True Tag -> Predicted Tag')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_pairs.png')
    plt.close()

def evaluate_hmm(tagger, test_data: List[List[Tuple[str, str]]]) -> Dict:
    """
    Evaluate HMM tagger using multiple metrics.
    Args:
        tagger: Trained HMM tagger
        test_data: List of test sequences
    Returns:
        Dictionary containing evaluation metrics
    """
    total = 0
    correct = 0
    all_true_tags = []
    all_predicted_tags = []
    tag_counts = Counter()
    reference_sequences = []
    candidate_sequences = []
    sequence_lengths = []

    for sentence in test_data:
        words = [w for w, _ in sentence]
        true_tags = [t for _, t in sentence]
        predicted_tags = [t for _, t in tagger.tag(words)]
        
        all_true_tags.extend(true_tags)
        all_predicted_tags.extend(predicted_tags)
        
        # Store sequences for BLEU and ROUGE
        reference_sequences.append(true_tags)
        candidate_sequences.append(predicted_tags)
        
        # Store sequence length
        sequence_lengths.append(len(true_tags))
        
        # Update tag counts
        tag_counts.update(true_tags)

        for pred, true in zip(predicted_tags, true_tags):
            total += 1
            if pred == true:
                correct += 1

    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Calculate sequence-level metrics
    seq_metrics = evaluate_sequence_metrics(reference_sequences, candidate_sequences)
    
    # Calculate precision, recall, and F1 for each tag
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_tags, all_predicted_tags, average=None, labels=sorted(set(all_true_tags))
    )
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_tags, all_predicted_tags, labels=sorted(set(all_true_tags)))
    
    # Calculate BLEU and ROUGE scores
    bleu_score = evaluate_bleu(reference_sequences, candidate_sequences)
    rouge_scores = evaluate_rouge(reference_sequences, candidate_sequences)
    
    # Analyze errors
    error_analysis = analyze_errors(all_true_tags, all_predicted_tags)
    
    # Plot error analysis
    plot_error_analysis(error_analysis['error_patterns'], error_analysis['confusion_pairs'])
    
    return {
        'accuracy': accuracy,
        'sequence_metrics': seq_metrics,
        'precision': dict(zip(sorted(set(all_true_tags)), precision)),
        'recall': dict(zip(sorted(set(all_true_tags)), recall)),
        'f1': dict(zip(sorted(set(all_true_tags)), f1)),
        'confusion_matrix': cm,
        'labels': sorted(set(all_true_tags)),
        'tag_counts': tag_counts,
        'bleu': bleu_score,
        'rouge': rouge_scores,
        'error_analysis': error_analysis,
        'sequence_lengths': sequence_lengths
    }

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = 'Confusion Matrix'):
    """Plot confusion matrix with labels."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_tag_distribution(tag_counts: Counter):
    """Plot distribution of tags in the dataset."""
    plt.figure(figsize=(12, 6))
    tags, counts = zip(*tag_counts.most_common())
    plt.bar(tags, counts)
    plt.title('Tag Distribution in Dataset')
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tag_distribution.png')
    plt.close()

def print_evaluation_metrics(metrics: Dict):
    """Print evaluation metrics in a readable format."""
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
    print(f"\nSequence-level Metrics:")
    print(f"Exact Match: {metrics['sequence_metrics']['exact_match']:.2%}")
    print(f"Partial Match: {metrics['sequence_metrics']['partial_match']:.2%}")
    
    print(f"\nBLEU Score: {metrics['bleu']:.4f}")
    print("\nROUGE Scores:")
    print(f"ROUGE-1: {metrics['rouge']['rouge-1']}")
    print(f"ROUGE-2: {metrics['rouge']['rouge-2']}")
    print(f"ROUGE-L: {metrics['rouge']['rouge-l']}")
    
    print("\nPer-tag metrics:")
    print(f"{'Tag':<15} {'Count':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 55)
    
    for tag in metrics['labels']:
        count = metrics['tag_counts'][tag]
        print(f"{tag:<15} {count:<10} {metrics['precision'][tag]:<10.2%} {metrics['recall'][tag]:<10.2%} {metrics['f1'][tag]:<10.2%}")
    
    print("\nTop Error Patterns:")
    error_patterns = sorted(metrics['error_analysis']['error_patterns'].items(), 
                          key=lambda x: x[1], reverse=True)[:5]
    for (true, pred), count in error_patterns:
        print(f"{true} -> {pred}: {count}")
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], metrics['labels'])
    
    # Plot sequence length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['sequence_lengths'], bins=20)
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig('sequence_lengths.png')
    plt.close()

if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess_data('old_finance_articles_triplets.csv')
    
    if not data:
        print("Error: No valid data found in the file.")
        exit(1)
    
    # Split data into train and test sets
    random.shuffle(data)
    split_index = int(0.8 * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    print(f"Total sequences: {len(data)}")
    print(f"Training sequences: {len(train_data)}")
    print(f"Test sequences: {len(test_data)}")
    
    # Train the HMM tagger
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_data)

    # Evaluate the tagger
    metrics = evaluate_hmm(tagger, test_data)
    print_evaluation_metrics(metrics)