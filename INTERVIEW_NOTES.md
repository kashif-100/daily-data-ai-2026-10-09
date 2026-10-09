# INTERVIEW_NOTES.md — Customer Churn Predictor (Day 2)

Interview Q&A prep. Every answer is grounded in what this project actually
does — if asked something deeper, answer from this file, don't invent.

## 30-second pitch
I built a telecom customer churn predictor on 7,000 synthetic customers. I did
pandas EDA to find who churns, then trained a RandomForest classifier to score
each customer's churn risk — it catches 58% of churners with a 53% hit rate,
about 1.5× the base rate. An auto-generated insights module turns the metrics
into a plain-English retention report, including a ranked list of 2,657
high-risk customers for the retention team to contact.

## Project walkthrough (end-to-end)
**Data:** I generated the data with a script using a fixed seed (42), so it's
fully reproducible. 7,000 rows, 20 features — demographics, services, contract
type, billing — plus a churn label produced by a logistic-style rule where
month-to-month contracts, short tenure, high monthly charges and missing tech
support raise churn probability. That gave a realistic 34% churn rate.
**EDA:** I measured churn by segment: 46.5% for month-to-month vs 15.3% for
two-year contracts, 46.2% for fiber-optic vs 26.0% for DSL, and 37.5% in the
first 6 months dropping to 27.9% after two years. **Modelling:** I one-hot
encoded the categoricals (24 features), stratified an 80/20 train/test split,
and trained a 200-tree RandomForest with balanced class weights. **Outputs:**
feature-importance and churn-breakdown charts, a metrics JSON, and an
auto-written narrative report.
**Why synthetic data:** a real customer dataset wasn't available, and a seeded
generator lets anyone reproduce the exact same data and results — the README
documents this honestly, and interviewers get a verifiable pipeline instead of
a black-box dataset.

## Technical deep-dives

**Q: Why RandomForest for churn?**
RandomForest fits churn data well: the drivers interact (high bill *and* no
tech support is worse than either alone), and tree ensembles capture those
non-linear interactions without me hand-engineering them. It's robust to the
mix of numeric and one-hot categorical features, and it gives a free feature
importance ranking that business users understand. Limitation: it's not the
most calibrated probability estimator out of the box, and 200 trees is slower
than a linear model — fine at this scale, wasteful at millions of rows.

**Q: How did you handle class imbalance?**
The classes aren't severely imbalanced (34% churn), but I still used
`class_weight="balanced"` so the model doesn't default to predicting the
majority class, and I stratified the train/test split so both sets keep the
same 34% churn share. The honest test is the confusion matrix: precision 0.53
and recall 0.58 on the churn class, which I report alongside accuracy because
accuracy alone (0.68) would hide the trade-off.

**Q: Why those metrics — precision, recall, F1, ROC-AUC?**
In retention, a false positive means wasting an offer on a loyal customer and
a false negative means losing a customer you could have saved. Precision
(0.53) answers "when we flag someone, how often are we right?", recall (0.58)
answers "of the people who actually churned, how many did we catch?". F1
(0.55) balances the two, and ROC-AUC (0.72) measures ranking quality across
all thresholds — decent, not great. I never tuned the threshold (fixed at
0.5); a real deployment would pick it based on the cost ratio of a lost
customer vs a wasted offer.

**Q: What did feature importance tell you?**
Monthly charges (0.19), total charges (0.15) and tenure (0.12) topped the
list, followed by contract type and fiber-optic service — the same story the
EDA told, which is a good consistency check. One honest caveat: monthly and
total charges are correlated (total ≈ monthly × tenure), so importance is
split between them; correlated features dilute each other's importance
scores. If I wanted cleaner attribution I'd use permutation importance or
SHAP.

**Q: Any overfitting concerns?**
I evaluated on a held-out test set the model never saw, which is the honest
check. I didn't do cross-validation or a validation split for tuning — I
didn't tune anything, so there's no tuning overfit, but I also don't know how
stable the metrics are across splits. The tree depth is uncapped, which is
normally risky, but I set `min_samples_leaf=2` and 200 trees, and the
test metrics are close to what I'd expect from the EDA — no red flags.

## Data-analyst thinking
**KPIs I tracked:** overall churn rate (34%), churn by contract/internet/
tenure/payment segments, average bill and tenure split by churned vs stayed,
plus the model's precision/recall on the churn class. **Key insight:** the
story isn't "random people leave" — it's month-to-month customers paying the
highest bills with no tech support, especially in their first six months.
**From analysis to recommendations:** each recommendation maps to a measured
gap — convert month-to-month to annual (46.5% vs 15.3% churn), add a
30/60/90-day onboarding (37.5% first-half-year churn), bundle tech support
with fiber (churners pay Rs 65 vs Rs 56 but 75.2% lack support). The flagged
customer list gives the retention team a place to start today.

## The "AI integrated" part
Here "AI integrated" means the reporting layer, not a chatbot: `ai_insights.py`
reads the computed metrics JSON and generates a plain-English narrative
report with findings, explanations and a retention plan — no LLM involved, no
API calls. **Trade-offs vs alternatives:** a template-based generator is
deterministic, free, always grounded in the real numbers, and can't
hallucinate — but it can't answer follow-up questions or adapt its narrative
like an LLM could. For a daily automated pipeline, deterministic and
hallucination-free wins; an LLM summary would add flexibility at the cost of
accuracy risk and API dependency.

## With more time I would…
Add cross-validation to check metric stability, tune the decision threshold
by retention-offer cost, and compare against logistic regression and
gradient boosting to justify the RandomForest choice. I'd try permutation
importance for cleaner feature attribution, add a churn-risk dashboard
segmented by contract and tenure, and simulate the revenue impact of the
retention plan (e.g. "saving 10% of flagged customers is worth Rs X").
