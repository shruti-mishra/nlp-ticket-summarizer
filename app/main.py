import streamlit as st
import pandas as pd
import io
from summariser import summarise_all
from evaluator import evaluate_summary, find_best_model

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NLP Ticket Summariser",
    page_icon="📝",
    layout="wide"
)

st.title("📝 NLP Ticket Summariser")
st.caption("Compare T5, BART, and BERT summarisation models on help desk tickets")

# ── Sidebar: File upload ──────────────────────────────────────────────────────

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or TXT file",
    type=["csv", "txt"],
    help="CSV: one ticket per row. TXT: one ticket per line."
)

# ── Load sample data by default ───────────────────────────────────────────────

sample_tickets = [
    "User cannot login to VPN since this morning. Tried restarting laptop and reinstalling the client but issue persists. Getting error code 403.",
    "Email sync not working on Outlook. Shows endless loading. Restarted Outlook and cleared cache but still broken.",
    "Cannot connect to shared drive at //fileserver/projects. Permission denied error. Need access urgently.",
    "Printer keeps going offline. Can print once then it disconnects. Tried restarting printer and reinstalling drivers.",
    "Git clone failing with SSL certificate error. Certificate verification failed. Cannot pull code from repo.",
]

if uploaded_file is not None:
    # Parse uploaded file
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        # Assume first column is the ticket text
        tickets = df.iloc[:, 0].tolist()
    else:  # .txt
        content = uploaded_file.read().decode("utf-8")
        tickets = [line.strip() for line in content.split('\n') if line.strip()]
else:
    # Use sample tickets
    tickets = sample_tickets
    st.info(f"📌 Using {len(tickets)} sample help desk tickets. Upload your own CSV or TXT file to use your data.")

# ── Process tickets ───────────────────────────────────────────────────────────

st.header("Summaries by Model")

# Create a placeholder for the progress bar
progress_placeholder = st.empty()
results_placeholder = st.empty()

# Process each ticket
all_results = []

for idx, ticket in enumerate(tickets):
    # Update progress
    progress = (idx + 1) / len(tickets)
    progress_placeholder.progress(progress, text=f"Processing {idx + 1}/{len(tickets)} tickets...")
    
    # Run all three models
    summaries = summarise_all(ticket)
    
    # Evaluate against the original ticket (as reference)
    scores = {}
    for model_name, summary_text in summaries.items():
        # Use first 100 chars of original ticket as reference for scoring
        ref_text = ticket[:200]  # Limit reference length
        model_scores = evaluate_summary(ref_text, summary_text)
        scores[model_name] = model_scores
    
    # Find best model for this ticket
    best_model = find_best_model(scores)
    
    all_results.append({
        'ticket_idx': idx + 1,
        'original': ticket,
        'summaries': summaries,
        'scores': scores,
        'best_model': best_model
    })

# Clear progress bar
progress_placeholder.empty()

# ── Display results in tabs ───────────────────────────────────────────────────

st.subheader(f"Results ({len(tickets)} tickets)")

# Create tabs, one per ticket
tabs = st.tabs([f"Ticket {r['ticket_idx']}" for r in all_results])

for tab, result in zip(tabs, all_results):
    with tab:
        # Show original ticket
        with st.expander("📌 Original Ticket", expanded=False):
            st.text(result['original'])
        
        # Show summaries in three columns
        col1, col2, col3 = st.columns(3)
        
        models = ['T5 (abstractive)', 'BART (abstractive)', 'BERT (extractive)']
        cols = [col1, col2, col3]
        
        for model_name, col in zip(models, cols):
            with col:
                # Model header
                is_best = "⭐ " if result['best_model'] == model_name else ""
                st.markdown(f"**{is_best}{model_name}**")
                
                # Summary text
                st.text_area(
                    "Summary",
                    value=result['summaries'][model_name],
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"summary_{result['ticket_idx']}_{model_name.split()[0]}"
                )
                
                # ROUGE scores
                st.markdown("**ROUGE Scores**")
                scores_dict = result['scores'][model_name]
                for metric, score in scores_dict.items():
                    col_metric, col_score = st.columns([2, 1])
                    with col_metric:
                        st.caption(metric)
                    with col_score:
                        st.caption(f"**{score}**")

# ── Summary statistics ────────────────────────────────────────────────────────

st.header("Model Comparison")

# Calculate average ROUGE scores across all tickets
model_avg_scores = {
    'T5 (abstractive)': {'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []},
    'BART (abstractive)': {'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []},
    'BERT (extractive)': {'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []},
}

for result in all_results:
    for model_name, scores in result['scores'].items():
        for metric, score in scores.items():
            model_avg_scores[model_name][metric].append(score)

# Display comparison table
comparison_data = []
for model_name, metrics in model_avg_scores.items():
    comparison_data.append({
        'Model': model_name,
        'Avg ROUGE-1': round(sum(metrics['ROUGE-1']) / len(metrics['ROUGE-1']), 3),
        'Avg ROUGE-2': round(sum(metrics['ROUGE-2']) / len(metrics['ROUGE-2']), 3),
        'Avg ROUGE-L': round(sum(metrics['ROUGE-L']) / len(metrics['ROUGE-L']), 3),
    })

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

st.markdown("""
**Model notes:**
- **T5 (abstractive)**: Generates entirely new sentences. Best for concise summaries.
- **BART (abstractive)**: Also generative but handles noisy text better than T5.
- **BERT (extractive)**: Selects existing sentences. Always faithful to the original.
""")

# ── Download results ──────────────────────────────────────────────────────────

st.header("Export Results")

# Create downloadable CSV
export_data = []
for result in all_results:
    row = {
        'Original Ticket': result['original'][:100],  # First 100 chars
    }
    for model_name, summary in result['summaries'].items():
        row[f'{model_name} Summary'] = summary
        for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
            row[f'{model_name} {metric}'] = result['scores'][model_name][metric]
    row['Best Model'] = result['best_model']
    export_data.append(row)

export_df = pd.DataFrame(export_data)
csv = export_df.to_csv(index=False)

st.download_button(
    label="📥 Download Results as CSV",
    data=csv,
    file_name="summarisation_results.csv",
    mime="text/csv"
)