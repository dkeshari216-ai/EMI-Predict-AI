
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1. DATASET GENERATION
# ─────────────────────────────────────────────

EMI_SCENARIOS = {
    "E-commerce": {"amount_range": (10000, 200000),  "tenure_range": (3, 24)},
    "Home_Appliances": {"amount_range": (20000, 300000),  "tenure_range": (6, 36)},
    "Vehicle":      {"amount_range": (80000, 1500000), "tenure_range": (12, 84)},
    "Personal_Loan":{"amount_range": (50000, 1000000), "tenure_range": (12, 60)},
    "Education":    {"amount_range": (50000, 500000),  "tenure_range": (6, 48)},
}

def generate_dataset(n_per_scenario: int = 80000) -> pd.DataFrame:
    """Generate realistic financial dataset with 400K records."""
    dfs = []
    for scenario, cfg in EMI_SCENARIOS.items():
        n = n_per_scenario
        age               = np.random.randint(25, 61, n)
        gender            = np.random.choice(["Male", "Female"], n, p=[0.6, 0.4])
        marital_status    = np.random.choice(["Single", "Married"], n, p=[0.4, 0.6])
        education         = np.random.choice(
            ["High School", "Graduate", "Post Graduate", "Professional"], n,
            p=[0.15, 0.40, 0.30, 0.15])
        employment_type   = np.random.choice(
            ["Private", "Government", "Self-employed"], n, p=[0.55, 0.25, 0.20])
        years_of_employment = np.clip(
            np.random.normal(loc=age - 22, scale=4, size=n).astype(int), 1, 35)
        company_type      = np.random.choice(
            ["Large", "Medium", "Small", "Startup"], n, p=[0.30, 0.30, 0.25, 0.15])
        house_type        = np.random.choice(
            ["Rented", "Own", "Family"], n, p=[0.40, 0.35, 0.25])
        monthly_rent      = np.where(
            house_type == "Rented", np.random.randint(3000, 30001, n), 0)
        family_size       = np.random.randint(1, 7, n)
        dependents        = np.clip(
            np.random.randint(0, family_size), 0, family_size - 1)

        # income correlated with education & employment
        edu_mult = {"High School": 0.7, "Graduate": 1.0,
                    "Post Graduate": 1.3, "Professional": 1.6}
        emp_mult = {"Private": 1.0, "Government": 1.1, "Self-employed": 1.2}
        base_salary = np.random.randint(15000, 200001, n)
        monthly_salary = (
            base_salary
            * np.array([edu_mult[e] for e in education])
            * np.array([emp_mult[e] for e in employment_type])
        ).astype(int)
        monthly_salary = np.clip(monthly_salary, 15000, 300000)

        school_fees       = np.where(dependents > 0,
            np.random.randint(500, 5001, n) * dependents, 0)
        college_fees      = np.where(dependents > 0,
            np.random.randint(0, 10001, n), 0)
        travel_expenses   = np.random.randint(500, 8001, n)
        groceries_util    = (monthly_salary * np.random.uniform(0.10, 0.25, n)).astype(int)
        other_expenses    = np.random.randint(500, 10001, n)

        existing_loans    = np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])
        current_emi_amount= np.where(
            existing_loans == "Yes", np.random.randint(1000, 20001, n), 0)
        credit_score      = np.clip(
            np.random.normal(650, 80, n).astype(int), 300, 850)
        bank_balance      = (monthly_salary * np.random.uniform(0.5, 6.0, n)).astype(int)
        emergency_fund    = (monthly_salary * np.random.uniform(0.0, 3.0, n)).astype(int)

        req_min, req_max  = cfg["amount_range"]
        requested_amount  = np.random.randint(req_min, req_max + 1, n)
        ten_min, ten_max  = cfg["tenure_range"]
        requested_tenure  = np.random.choice(range(ten_min, ten_max + 1, 6) or
                                              range(ten_min, ten_max + 1), n)

        df = pd.DataFrame({
            "age": age, "gender": gender, "marital_status": marital_status,
            "education": education, "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type, "house_type": house_type,
            "monthly_rent": monthly_rent, "family_size": family_size,
            "dependents": dependents, "school_fees": school_fees,
            "college_fees": college_fees, "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_util,
            "other_monthly_expenses": other_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score, "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
        })
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    data = _add_targets(data)
    return data


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive classification and regression targets from financial features."""
    total_expenses = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"]
        + df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    disposable = df["monthly_salary"] - total_expenses
    dti = total_expenses / df["monthly_salary"].clip(lower=1)
    monthly_rate = 0.12 / 12
    tenure = df["requested_tenure"].clip(lower=1)
    # approximate EMI for requested loan
    req_emi = (df["requested_amount"] * monthly_rate
               * (1 + monthly_rate) ** tenure) / ((1 + monthly_rate) ** tenure - 1)

    # max safe EMI = 40% of disposable income, min 500
    max_emi = (disposable * 0.40).clip(lower=500, upper=50000).round(0)
    df["max_monthly_emi"] = max_emi.astype(int)

    # classification logic
    conditions = [
        (df["credit_score"] >= 700) & (dti < 0.40) & (disposable > req_emi * 1.2),
        (df["credit_score"] >= 600) & (dti < 0.55) & (disposable > req_emi * 0.8),
    ]
    choices = ["Eligible", "High_Risk"]
    df["emi_eligibility"] = np.select(conditions, choices, default="Not_Eligible")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

CATEGORICAL_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]
NUMERICAL_COLS = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and outlier clipping."""
    df = df.copy()
    df.drop_duplicates(inplace=True)
    # fill numeric NaNs with median
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    # fill categorical NaNs with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    # clip obvious outliers
    df["monthly_salary"]    = df["monthly_salary"].clip(15000, 300000)
    df["credit_score"]      = df["credit_score"].clip(300, 850)
    df["requested_amount"]  = df["requested_amount"].clip(lower=5000)
    df["requested_tenure"]  = df["requested_tenure"].clip(lower=1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived financial ratios and risk features."""
    df = df.copy()
    total_exp = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"]
        + df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    df["total_monthly_expenses"] = total_exp
    df["disposable_income"]      = df["monthly_salary"] - total_exp
    df["debt_to_income_ratio"]   = (total_exp / df["monthly_salary"].clip(lower=1)).round(4)
    df["expense_to_income_ratio"]= (total_exp / df["monthly_salary"].clip(lower=1)).round(4)
    df["savings_ratio"]          = (df["bank_balance"] / df["monthly_salary"].clip(lower=1)).round(4)
    df["emergency_fund_months"]  = (df["emergency_fund"] / df["monthly_salary"].clip(lower=1)).round(2)

    # requested EMI affordability
    r = 0.12 / 12
    t = df["requested_tenure"].clip(lower=1)
    req_emi = (df["requested_amount"] * r * (1 + r) ** t) / ((1 + r) ** t - 1)
    df["requested_emi_estimate"]   = req_emi.round(0).astype(int)
    df["emi_to_income_ratio"]      = (req_emi / df["monthly_salary"].clip(lower=1)).round(4)
    df["affordability_score"]      = (df["disposable_income"] / req_emi.clip(lower=1)).round(4)

    # risk scoring
    df["credit_risk_score"]    = (df["credit_score"] / 850 * 100).round(2)
    df["employment_stability"] = df["years_of_employment"].clip(0, 20) / 20 * 100
    df["financial_health_index"]= (
        df["credit_risk_score"] * 0.4
        + (1 - df["debt_to_income_ratio"]) * 100 * 0.3
        + df["savings_ratio"].clip(0, 5) / 5 * 100 * 0.3
    ).round(2)

    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True,
                     encoders: dict = None, scaler: StandardScaler = None):
    """Label-encode categoricals and standard-scale numericals."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    feature_cols = [c for c in df.columns
                    if c not in ("emi_eligibility", "max_monthly_emi")]
    num_cols_present = [c for c in NUMERICAL_COLS + [
        "total_monthly_expenses", "disposable_income", "debt_to_income_ratio",
        "expense_to_income_ratio", "savings_ratio", "emergency_fund_months",
        "requested_emi_estimate", "emi_to_income_ratio", "affordability_score",
        "credit_risk_score", "employment_stability", "financial_health_index",
    ] if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[num_cols_present] = scaler.fit_transform(df[num_cols_present])
    else:
        df[num_cols_present] = scaler.transform(df[num_cols_present])

    return df, encoders, scaler


def prepare_data(df: pd.DataFrame):
    """Full pipeline: clean → engineer → encode → split."""
    df = clean_data(df)
    df = engineer_features(df)

    # encode targets
    eli_map = {"Eligible": 0, "High_Risk": 1, "Not_Eligible": 2}
    df["emi_eligibility_enc"] = df["emi_eligibility"].map(eli_map)

    target_clf = df["emi_eligibility_enc"]
    target_reg = df["max_monthly_emi"]

    feature_df = df.drop(columns=["emi_eligibility", "emi_eligibility_enc", "max_monthly_emi"])
    feature_df, encoders, scaler = encode_and_scale(feature_df)

    X_train, X_temp, y_clf_train, y_clf_temp, y_reg_train, y_reg_temp = train_test_split(
        feature_df, target_clf, target_reg, test_size=0.30, random_state=RANDOM_STATE, stratify=target_clf)
    X_val, X_test, y_clf_val, y_clf_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_clf_temp, y_reg_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_clf_temp)

    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_clf_train": y_clf_train, "y_clf_val": y_clf_val, "y_clf_test": y_clf_test,
        "y_reg_train": y_reg_train, "y_reg_val": y_reg_val, "y_reg_test": y_reg_test,
    }
    return splits, encoders, scaler, df


if __name__ == "__main__":
    print("Generating 400K records …")
    raw = generate_dataset()
    print(f"Dataset shape: {raw.shape}")
    print(raw["emi_eligibility"].value_counts())
    splits, enc, sc, processed = prepare_data(raw)
    print(f"Train: {splits['X_train'].shape}  Val: {splits['X_val'].shape}  Test: {splits['X_test'].shape}")
    raw.to_csv("emi_dataset.csv", index=False)
    print("Saved emi_dataset.csv")