"""Generate synthetic telecom customer churn data (seeded, reproducible).

Models realistic churn behaviour:
- month-to-month contracts churn far more than 1- or 2-year contracts
- short-tenure and high-monthly-charge customers are riskier
- fiber-optic customers pay more and churn slightly more
- add-on services (tech support, online security) reduce churn
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 7000

rng = np.random.default_rng(RANDOM_SEED)

# --- demographics / account features -------------------------------------
gender = rng.choice(["Male", "Female"], N_CUSTOMERS)
senior = rng.choice([0, 1], N_CUSTOMERS, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.52, 0.48])
dependents = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.30, 0.70])

# tenure: many new customers, fewer very long ones (clipped at 72 months)
tenure = np.clip(rng.exponential(scale=18, size=N_CUSTOMERS).astype(int), 1, 72)

# --- services -------------------------------------------------------------
phone_service = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.90, 0.10])
multiple_lines = np.where(phone_service == "Yes",
                          rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.42, 0.58]),
                          "No phone service")
internet_service = rng.choice(["DSL", "Fiber optic", "No"],
                              N_CUSTOMERS, p=[0.34, 0.44, 0.22])

def add_on(p_yes):
    return rng.choice(["Yes", "No"], N_CUSTOMERS, p=[p_yes, 1 - p_yes])

online_security = add_on(0.30)
online_backup = add_on(0.34)
device_protection = add_on(0.34)
tech_support = add_on(0.29)
streaming_tv = add_on(0.38)
streaming_movies = add_on(0.39)

# --- contract & billing ----------------------------------------------------
contract = rng.choice(["Month-to-month", "One year", "Two year"],
                      N_CUSTOMERS, p=[0.55, 0.24, 0.21])
paperless_billing = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.60, 0.40])
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)",
     "Credit card (automatic)"], N_CUSTOMERS, p=[0.34, 0.19, 0.22, 0.25])

# monthly charges: base + fiber premium + add-on stack, with noise
base = 25 + 45 * (internet_service == "Fiber optic") + 10 * (internet_service == "DSL")
addons = (online_security == "Yes") * 4 + (online_backup == "Yes") * 3 \
    + (device_protection == "Yes") * 4 + (tech_support == "Yes") * 5 \
    + (streaming_tv == "Yes") * 7 + (streaming_movies == "Yes") * 7
monthly_charges = np.round(base + addons + rng.normal(0, 5, N_CUSTOMERS), 2).clip(18, 120)
total_charges = np.round(monthly_charges * tenure + rng.normal(0, 30, N_CUSTOMERS), 2).clip(18)

# --- churn label: logistic-style model of the drivers ----------------------
logit = (
    -1.6
    + 1.15 * (contract == "Month-to-month")
    - 0.60 * (contract == "Two year")
    + 0.012 * (60 - tenure)                    # short tenure = riskier
    + 0.018 * (monthly_charges - 65)           # high bill = riskier
    + 0.35 * (internet_service == "Fiber optic")
    - 0.45 * (tech_support == "Yes")
    - 0.30 * (online_security == "Yes")
    - 0.25 * (partner == "Yes")
    + 0.20 * (payment_method == "Electronic check")
)
prob = 1 / (1 + np.exp(-logit))
churn = np.where(rng.random(N_CUSTOMERS) < prob, "Yes", "No")

df = pd.DataFrame({
    "customer_id": [f"C{100000 + i}" for i in range(N_CUSTOMERS)],
    "gender": gender,
    "senior_citizen": senior,
    "partner": partner,
    "dependents": dependents,
    "tenure_months": tenure,
    "phone_service": phone_service,
    "multiple_lines": multiple_lines,
    "internet_service": internet_service,
    "online_security": online_security,
    "online_backup": online_backup,
    "device_protection": device_protection,
    "tech_support": tech_support,
    "streaming_tv": streaming_tv,
    "streaming_movies": streaming_movies,
    "contract": contract,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "churn": churn,
})

df.to_csv("data/churn_data.csv", index=False)
print(f"Saved {len(df):,} customers → data/churn_data.csv "
      f"(churn rate: {(df.churn == 'Yes').mean():.1%})")
