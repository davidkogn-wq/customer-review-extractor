import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Review & Complaint Extractor",
    page_icon="📊",
    layout="wide"
)

# 2. Load Models (Cached so they don't reload on every click)
@st.cache_resource
def load_models():
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    # Lightweight zero-shot model (~260MB) for custom business topics
    topic_pipe = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli"
    )
    return sentiment_pipe, topic_pipe

sentiment_pipe, topic_pipe = load_models()

# 3. Sample Data Generator for One-Click Testing
def get_sample_data():
    data = [
        {"review_id": 101, "date": "2026-09-01", "product": "Wireless Earbuds", "review_text": "Battery dies after 20 minutes of use! Completely defective and won't hold a charge."},
        {"review_id": 102, "date": "2026-09-02", "product": "Wireless Earbuds", "review_text": "Sound quality is crystal clear and the bass is punchy. Best purchase this year!"},
        {"review_id": 103, "date": "2026-09-03", "product": "Running Jacket", "review_text": "Ordered a Large and it fits like an Extra Small. The sizing chart is completely inaccurate."},
        {"review_id": 104, "date": "2026-09-04", "product": "Smart Watch", "review_text": "Package arrived 3 weeks late and the box was crushed. Terrible shipping experience."},
        {"review_id": 105, "date": "2026-09-05", "product": "Smart Watch", "review_text": "Customer service ignored my three emails asking for a refund. Worst support ever!"},
        {"review_id": 106, "date": "2026-09-06", "product": "Wireless Earbuds", "review_text": "Left earbud stopped working on day two. Cheap plastic build quality."},
        {"review_id": 107, "date": "2026-09-07", "product": "Running Jacket", "review_text": "Love the material! Kept me dry during a heavy downpour and looks great."},
        {"review_id": 108, "date": "2026-09-08", "product": "Smart Watch", "review_text": "The screen froze and now it won't turn on at all. Total waste of money, need a replacement ASAP."},
        {"review_id": 109, "date": "2026-09-09", "product": "Running Jacket", "review_text": "Still waiting for my package to arrive. Tracking hasn't updated in 10 days."},
        {"review_id": 110, "date": "2026-09-10", "product": "Wireless Earbuds", "review_text": "Super comfortable fit and pairs instantly with my phone."}
    ]
    return pd.DataFrame(data)

# 4. Sidebar Controls & Custom Categories
st.sidebar.header("⚙️ Configuration")
default_topics = "Defective Battery, Sizing Issue, Shipping Delay, Customer Service, Build Quality"
custom_topics_input = st.sidebar.text_area(
    "Complaint Categories (comma-separated):",
    value=default_topics,
    help="The AI will group negative reviews into these exact buckets."
)
candidate_labels = [t.strip() for t in custom_topics_input.split(",") if t.strip()]

# 5. Main UI Header & Ingestion
st.title("📊 Customer Review & Complaint Extractor")
st.markdown("Upload a CSV of product reviews to automatically score sentiment, group complaints by topic, and flag urgent issues.")

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV (must contain a 'review_text' column)", type=["csv"])

with col_sample:
    st.write("") # Spacing
    st.write("")
    load_sample = st.button("🚀 Load Sample Amazon Data", use_container_width=True)

# Store dataframe in session state so it persists across clicks
if load_sample:
    st.session_state["df"] = get_sample_data()
elif uploaded_file is not None:
    st.session_state["df"] = pd.read_csv(uploaded_file)

# 6. Processing & Dashboard Display
if "df" in st.session_state:
    df = st.session_state["df"].copy()

    if "review_text" not in df.columns:
        st.error("Error: Your CSV must contain a column named `review_text`.")
        st.stop()

    # Clean empty rows
    df = df.dropna(subset=["review_text"]).reset_index(drop=True)

    with st.spinner("Analyzing sentiment and extracting complaint topics..."):
        texts = df["review_text"].astype(str).tolist()

        # A. Sentiment Scoring (-1 to +1)
        sent_results = sentiment_pipe(texts, truncation=True, max_length=512)
        sentiments = []
        scores = []

        for r in sent_results:
            label = r["label"] # POSITIVE or NEGATIVE
            conf = r["score"]
            # Convert confidence into a -1.0 to +1.0 scale
            signed_score = conf if label == "POSITIVE" else -conf
            sentiments.append(label.capitalize())
            scores.append(round(signed_score, 4))

        df["Sentiment"] = sentiments
        df["Sentiment_Score"] = scores

        # B. Topic Extraction (Only run on Negative reviews to save compute time)
        topics = []
        for idx, row in df.iterrows():
            if row["Sentiment"] == "Negative" and candidate_labels:
                res = topic_pipe(row["review_text"], candidate_labels, truncation=True)
                topics.append(res["labels"][0]) # Top predicted category
            else:
                topics.append("N/A (Positive)")
        df["Primary_Topic"] = topics

    # --- KPI Metrics Row ---
    total_reviews = len(df)
    neg_df = df[df["Sentiment"] == "Negative"]
    neg_count = len(neg_df)
    pos_count = total_reviews - neg_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Reviews Analyzed", total_reviews)
    m2.metric("Positive Reviews", f"{pos_count} ({round(pos_count/total_reviews*100)}%)")
    m3.metric("Negative / Urgent Reviews", f"{neg_count} ({round(neg_count/total_reviews*100)}%)")

    st.divider()

    # --- Charts Row ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment Breakdown")
        fig_sent = px.pie(
            df,
            names="Sentiment",
            color="Sentiment",
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            hole=0.4
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with c2:
        st.subheader("Top Complaint Topics")
        if neg_count > 0:
            topic_counts = neg_df["Primary_Topic"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]
            fig_topics = px.bar(
                topic_counts,
                x="Count",
                y="Topic",
                orientation="h",
                color="Count",
                color_continuous_scale="Reds"
            )
            fig_topics.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_topics, use_container_width=True)
        else:
            st.info("No negative reviews found to categorize!")

    st.divider()

    # --- Urgent Triage Table ---
    st.subheader("🚨 Urgent Triage: Angriest Reviews Needing Immediate Reply")
    if neg_count > 0:
        # Sort by most negative score first (-0.9999 is angriest)
        urgent_df = neg_df.sort_values(by="Sentiment_Score", ascending=True)
        display_cols = [c for c in ["review_id", "date", "product", "Primary_Topic", "Sentiment_Score", "review_text"] if c in urgent_df.columns]
        st.dataframe(urgent_df[display_cols], use_container_width=True, hide_index=True)

        # Export button for business owners
        csv_export = urgent_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Urgent Triage Report (CSV)",
            data=csv_export,
            file_name="urgent_customer_complaints.csv",
            mime="text/csv"
        )
    else:
        st.success("All clear! No negative complaints detected.")
