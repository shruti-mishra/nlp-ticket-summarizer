import torch
import numpy as np
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    BertTokenizer, BertModel
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model cache ───────────────────────────────────────────────────────────────

_t5_model = None
_t5_tokenizer = None
_bart_model = None
_bart_tokenizer = None
_bert_tokenizer = None
_bert_model = None


def _get_t5():
    global _t5_model, _t5_tokenizer
    if _t5_model is None:
        _t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        _t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
        _t5_model.eval()
    return _t5_model, _t5_tokenizer


def _get_bart():
    global _bart_model, _bart_tokenizer
    if _bart_model is None:
        _bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        _bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(DEVICE)
        _bart_model.eval()
    return _bart_model, _bart_tokenizer


def _get_bert():
    global _bert_tokenizer, _bert_model
    if _bert_model is None:
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
        _bert_model.eval()
    return _bert_tokenizer, _bert_model


# ── Summarisation functions ───────────────────────────────────────────────────

def summarise_t5(text: str, max_length: int = 60, min_length: int = 20) -> str:
    """Abstractive summarisation using T5-small."""
    try:
        model, tokenizer = _get_t5()
        input_text = "summarize: " + text.strip()
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    except Exception as e:
        return f"T5 error: {str(e)}"


def summarise_bart(text: str, max_length: int = 60, min_length: int = 20) -> str:
    """Abstractive summarisation using BART-base."""
    try:
        model, tokenizer = _get_bart()
        inputs = tokenizer(text.strip(), return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()
    except Exception as e:
        return f"BART error: {str(e)}"


def summarise_bert(text: str, num_sentences: int = 2) -> str:
    """Extractive summarisation using BERT embeddings."""
    try:
        tokenizer, model = _get_bert()

        # Split into sentences
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        if not sentences:
            return text.strip()
        if len(sentences) <= num_sentences:
            return text.strip()

        # Embed each sentence
        def embed(sentence):
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            # Mean pool the last hidden state
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        sentence_vectors = [embed(s) for s in sentences]

        # Embed full document as reference
        doc_vector = embed(text[:512])

        # Score each sentence by cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

        scores = [cosine_similarity(sv, doc_vector) for sv in sentence_vectors]

        # Pick top N sentences, preserving original order
        top_indices = sorted(np.argsort(scores)[-num_sentences:].tolist())
        return ". ".join(sentences[i] for i in top_indices) + "."

    except Exception as e:
        return f"BERT error: {str(e)}"


def summarise_all(text: str) -> dict:
    """Run all three models. Returns dict used by the UI."""
    return {
        "T5 (abstractive)":   summarise_t5(text),
        "BART (abstractive)": summarise_bart(text),
        "BERT (extractive)":  summarise_bert(text),
    }