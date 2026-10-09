# Day 2 — Customer Churn Predictor 📉

Data-analyst project with an AI insights layer. Part of a 15-day daily series:
one data + AI project every day.

## What it does
1. **`generate_data.py`** — creates a synthetic telecom customer base
   (7,000 customers, seeded for reproducibility) with realistic churn
   behaviour: month-to-month contracts, short tenure, high bills and
   missing tech support make customers likelier to leave.
2. **`analysis.py`** — pandas EDA of churn by contract, internet service,
   tenure and payment method; then trains a scikit-learn **RandomForest**
   classifier (200 trees, balanced class weights) on a stratified 80/20
   split. Saves charts to `charts/` and metrics to `findings.json`.
3. **`ai_insights.py`** — the AI layer: converts the computed statistics into
   a plain-English narrative report (`AI_INSIGHTS.md`) with findings and a
   retention action plan. No human needed to read the spreadsheets.

## Headline findings
- **34%** of customers churned; **month-to-month contracts churn at 46.5%**
  vs **15.3%** on two-year contracts — commitment is the #1 lever
- **Fiber-optic customers churn at 46.2%** (highest bills) vs 26.0% on DSL
- Churners pay **Rs 65/month** on average vs **Rs 56** for stayers, and
  **75% of churners have no tech support**
- Model: accuracy 0.68, **precision 0.53, recall 0.58, ROC-AUC 0.72** —
  catches 58% of churners; flagged customers churn at 1.5× the base rate,
  so the list works as a prioritised retention queue, not gospel
- Top predictors: monthly charges, total charges, tenure, contract type

## How to run
```bash
pip install -r requirements.txt
python generate_data.py
python analysis.py
python ai_insights.py
```

> The dataset (`data/churn_data.csv`) is **generated** by `generate_data.py`
> (fixed seed 42) — it is not committed to the repo; run the generator to
> recreate it exactly.

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic churn dataset generator |
| `analysis.py` | EDA + RandomForest training + charts + `findings.json` |
| `ai_insights.py` | Auto-generated narrative report (`AI_INSIGHTS.md`) |
| `AI_INSIGHTS.md` | The AI-written churn report |
| `findings.json` | Machine-readable metrics behind the report |
| `INTERVIEW_NOTES.md` | Interview Q&A prep for this project |
| `charts/` | Churn-rate breakdowns, feature importance, confusion matrix |

*Day 2 of the 15-day daily-data-ai series by [kashif-100](https://github.com/kashif-100).*
