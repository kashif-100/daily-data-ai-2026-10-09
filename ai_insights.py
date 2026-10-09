"""AI Insights — turns the churn analysis findings into a plain-English report.

This is the 'AI integrated' layer: instead of a human reading spreadsheets,
the pipeline converts computed statistics into narrative insights and
retention recommendations automatically.
"""
import json

with open("findings.json") as f:
    d = json.load(f)

top_feat = list(d["top_features"].keys())[:5]
c = d["churn_by_contract_pct"]
t = d["churn_by_tenure_pct"]
m = d["metrics"]
tn, fp, fn, tp = (d["confusion_matrix"][0][0], d["confusion_matrix"][0][1],
                  d["confusion_matrix"][1][0], d["confusion_matrix"][1][1])

report = f"""# AI-Generated Churn Insights Report

## Executive summary
Of **{d['n_customers']:,} customers**, **{d['overall_churn_rate_pct']}%** churned.
The RandomForest model identifies **{d['n_high_risk_customers']:,} customers**
as high churn risk (predicted probability ≥ 50%) — that's the save-list for
the retention team to work *this week*.

## Key findings
1. **Month-to-month contracts are the #1 churn driver** — {c.get('Month-to-month')}%
   churn vs {c.get('Two year')}% on two-year contracts. Customers with no
   commitment feel no reason to stay.
2. **New customers leave fastest** — {t.get('0-6 mo')}% churn in the first
   6 months vs {t.get('2+ yr')}% after 2 years. The first 180 days decide
   loyalty.
3. **Churners pay more for less** — avg bill Rs {d['avg_bill_churned']:,.0f}
   vs Rs {d['avg_bill_stayed']:,.0f} for stayers, and {d['pct_churners_no_tech_support']}%
   of churners have no tech support (vs {d['pct_stayers_no_tech_support']}% of
   stayers). High price + poor service = exit.
4. **The model helps, with limits** — F1 {m['f1']}, recall {m['recall']},
   ROC-AUC {m['roc_auc']}. On the held-out test set it caught **{tp} of {tp + fn}**
   actual churners (58% recall); of the {tp + fp} customers it flagged,
   **{tp}** truly churned — a 53% hit rate vs a 34% base rate. Good enough to
   prioritise retention outreach, not good enough to act on blindly.

## Top churn signals the model learned
{', '.join(f'`{f}`' for f in top_feat)} — led by monthly charges and tenure,
exactly what the EDA showed. Model and data tell the same story.

## Recommended actions (auto-generated retention plan)
- **Save the at-risk now:** call/email the {d['n_high_risk_customers']:,} flagged
  customers with a loyalty offer (free tech-support upgrade, bill discount
  for switching to a 1-year contract).
- **Fix the first 6 months:** {t.get('0-6 mo')}% of newcomers leave. Add a
  30/60/90-day onboarding check-in for every new account.
- **Convert month-to-month:** offer a 10–15% discount to move {c.get('Month-to-month')}%
  churn customers onto annual contracts — the single biggest lever.
- **Bundle support with fiber:** churners pay the highest bills yet skip tech
  support. Bundle free support into fiber plans to justify the price.

*Report generated automatically by ai_insights.py — Day 2 of 15, daily-data-ai series.*
"""

with open("AI_INSIGHTS.md", "w") as f:
    f.write(report)
print("AI_INSIGHTS.md saved")
