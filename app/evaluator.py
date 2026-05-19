"""
ROUGE scoring for summarisation evaluation.
ROUGE measures overlap between generated and reference summaries.
- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap  
- ROUGE-L: longest common subsequence
"""

from rouge_score import rouge_scorer


def evaluate_summary(reference: str, hypothesis: str) -> dict:
    """
    Score a generated summary against a reference summary.
    
    Args:
        reference: ground truth summary (or the original text if no summary exists)
        hypothesis: generated summary to evaluate
    
    Returns:
        dict with ROUGE-1, ROUGE-2, ROUGE-L F-scores (0-1, rounded to 3 decimals)
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    
    return {
        'ROUGE-1': round(scores['rouge1'].fmeasure, 3),
        'ROUGE-2': round(scores['rouge2'].fmeasure, 3),
        'ROUGE-L': round(scores['rougeL'].fmeasure, 3),
    }


def evaluate_batch(references: list, hypotheses: list) -> list:
    """
    Score multiple summaries at once.
    
    Args:
        references: list of reference texts
        hypotheses: list of generated summaries (same length as references)
    
    Returns:
        list of dicts, each with ROUGE scores
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be same length")
    
    return [evaluate_summary(ref, hyp) for ref, hyp in zip(references, hypotheses)]


def find_best_model(scores_dict: dict) -> str:
    """
    Given a dict like {'T5': {'ROUGE-1': 0.5, ...}, 'BART': {...}},
    return the model name with highest average ROUGE score.
    Used by the UI to highlight the best performer.
    """
    best_model = None
    best_avg = -1
    
    for model_name, scores in scores_dict.items():
        avg_score = sum(scores.values()) / len(scores)
        if avg_score > best_avg:
            best_avg = avg_score
            best_model = model_name
    
    return best_model