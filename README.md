# customer-review-extractor
# AI Customer Review & Complaint Extractor

**[Click Here to Try the Live App]((https://customer-review-extractor-fhqsvrwg7i3bappqbyqo5hc.streamlit.app))**

An automated Business Intelligence (BI) web application that turns raw customer review spreadsheets into actionable product and support insights. Built with **Python**, **Streamlit**, **Plotly**, and **Hugging Face Transformers**.

## 🎯 The Business Problem
E-commerce store owners, SaaS founders, and marketing agencies spend hours manually reading through thousands of Amazon, Trustpilot, or Shopify reviews to figure out why customers are unhappy. Basic "positive/negative" sentiment analysis isn't enough—businesses need to know *what* is broken and *who* needs an immediate reply.

## 💡 How It Works
1. **Data Ingestion (`pandas`):** Accepts any custom CSV upload (or loads a built-in sample dataset with one click) and automatically cleans empty rows.
2. **Sentiment Scoring (`DistilBERT`):** Uses `distilbert-base-uncased-finetuned-sst-2-english` to score every review on a `-1.0` (angriest) to `+1.0` (happiest) scale.
3. **Smart Zero-Shot Topic Triage (`DistilBART`):** Filters only the negative reviews to cut compute time in half, then categorizes complaints into customizable business buckets (e.g., *Shipping Delay, Sizing Issue, Defective Battery, Broken Hardware*) using `valhalla/distilbart-mnli-12-3`—requiring zero model retraining.
4. **Actionable Export:** Ranks the angriest reviews at the top of an urgent triage table and provides a one-click CSV download for customer support teams.

## 🛠️ Tech Stack
* **Frontend & Dashboard:** Streamlit, Plotly Express
* **AI / NLP Pipeline:** Hugging Face `transformers`, PyTorch
* **Data Manipulation:** Pandas
