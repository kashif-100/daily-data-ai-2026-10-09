# AI-Generated Churn Insights Report

## Executive summary
Of **7,000 customers**, **34.01%** churned.
The RandomForest model identifies **2,657 customers**
as high churn risk (predicted probability ≥ 50%) — that's the save-list for
the retention team to work *this week*.

## Key findings
1. **Month-to-month contracts are the #1 churn driver** — 46.5%
   churn vs 15.3% on two-year contracts. Customers with no
   commitment feel no reason to stay.
2. **New customers leave fastest** — 37.5% churn in the first
   6 months vs 27.9% after 2 years. The first 180 days decide
   loyalty.
3. **Churners pay more for less** — avg bill Rs 65
   vs Rs 56 for stayers, and 75.2%
   of churners have no tech support (vs 70.0% of
   stayers). High price + poor service = exit.
4. **The model helps, with limits** — F1 0.5502, recall 0.5756,
   ROC-AUC 0.7225. On the held-out test set it caught **274 of 476**
   actual churners (58% recall); of the 520 customers it flagged,
   **274** truly churned — a 53% hit rate vs a 34% base rate. Good enough to
   prioritise retention outreach, not good enough to act on blindly.

## Top churn signals the model learned
`monthly_charges`, `total_charges`, `tenure_months`, `contract_Two year`, `contract_One year` — led by monthly charges and tenure,
exactly what the EDA showed. Model and data tell the same story.

## Recommended actions (auto-generated retention plan)
- **Save the at-risk now:** call/email the 2,657 flagged
  customers with a loyalty offer (free tech-support upgrade, bill discount
  for switching to a 1-year contract).
- **Fix the first 6 months:** 37.5% of newcomers leave. Add a
  30/60/90-day onboarding check-in for every new account.
- **Convert month-to-month:** offer a 10–15% discount to move 46.5%
  churn customers onto annual contracts — the single biggest lever.
- **Bundle support with fiber:** churners pay the highest bills yet skip tech
  support. Bundle free support into fiber plans to justify the price.

*Report generated automatically by ai_insights.py — Day 2 of 15, daily-data-ai series.*
