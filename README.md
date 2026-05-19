# 📝 NLP Ticket Summariser — Multi-Model Comparison

> Compare abstractive (T5, BART) and extractive (BERT) summarisation on IT help desk tickets. Built to understand model trade-offs in real-world scenarios.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![ROUGE](https://img.shields.io/badge/ROUGE-Evaluation-0467DF?style=flat)

---

## What it does

This app takes IT help desk tickets (messy, informal, variable length) and generates summaries using three different models in parallel. It then scores each summary using ROUGE metrics so you can see which approach works best for your data.

**Example:**
- Input: "User cannot login to VPN since this morning. Tried restarting..."
- T5: "user cannot login to VPN. tried restarting but issue persists."
- BART: "User cannot login to VPN since this morning. Tried restarting..."
- BERT: "User cannot login to VPN since this morning. Tried restarting..."

---

## Three models, three approaches

| Model | Type | Strength | Weakness |
|-------|------|----------|----------|
| **T5** | Abstractive | Generates concise summaries | Can hallucinate |
| **BART** | Abstractive | Handles noisy text well | Longer outputs |
| **BERT** | Extractive | Always faithful to original | Cannot rephrase |

---

## Run it locally

```bash
git clone git@github.com:shruti-mishra/nlp-ticket-summarizer.git
cd nlp-ticket-summarizer
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cd app
streamlit run main.py
```

Opens at `http://localhost:8501`. Uses 10 sample tickets by default or upload your own CSV/TXT.

---

## What I learned

**Model architecture trade-offs:** Abstractive models (T5, BART) paraphrase but risk hallucination. Extractive (BERT) is faithful but limited.

**Why BERT extractive works:** BERT embeddings capture semantic meaning. Sentences close to the document vector are representative.

**ROUGE metrics:** Measure n-gram overlap but don't guarantee readability. High ROUGE ≠ useful summary.

**Caching performance:** Loading 3× large transformers (1.2GB total) is slow. Global caching means each model loads once per session — huge win.

---

## Model comparison (10 sample tickets)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| T5 | 0.58 | 0.42 | 0.56 |
| BART | 0.72 | 0.51 | 0.70 |
| BERT | 0.81 | 0.68 | 0.80 |

BERT scores highest (extractive summaries share vocabulary). T5 is most concise. BART balances both.

---

## Tech stack

- **UI:** Streamlit
- **Models:** HuggingFace transformers (T5, BART, BERT)
- **Evaluation:** rouge-score
- **Data:** Pandas

---

## Author

**Shruti Mishra** — Building AI/ML tools. Focused on NLP, LLMs, and practical applications.

- [GitHub](https://github.com/shruti-mishra)
- [LinkedIn](https://www.linkedin.com/in/shruti-mishra-82b271107)