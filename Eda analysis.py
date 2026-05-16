
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
SNS_PALETTE = "husl"


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────
# 1. Basic statistics
# ─────────────────────────────────────────────

def basic_stats(df: pd.DataFrame) -> dict:
    stats = {
        "shape": df.shape,
        "missing": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "numeric_summary": df.describe().round(2),
        "eligibility_dist": df["emi_eligibility"].value_counts(),
        "scenario_dist": df["emi_scenario"].value_counts(),
    }
    return stats


# ─────────────────────────────────────────────
# 2. Distribution plots
# ─────────────────────────────────────────────

def plot_eligibility_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["emi_eligibility"].value_counts()
    colors = [PALETTE[k] for k in counts.index]
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors,
                startangle=140, textprops={"fontsize": 12})
    axes[0].set_title("EMI Eligibility Distribution", fontsize=14)
    sns.countplot(data=df, x="emi_eligibility", order=counts.index,
                  palette=PALETTE, ax=axes[1])
    axes[1].set_title("Eligibility Count by Category", fontsize=14)
    axes[1].set_xlabel("Eligibility Status")
    axes[1].set_ylabel("Count")
    for p in axes[1].patches:
        axes[1].annotate(f"{int(p.get_height()):,}",
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    return _save(fig, "01_eligibility_distribution.png")


def plot_scenario_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    scenario_eli = df.groupby(["emi_scenario", "emi_eligibility"]).size().unstack(fill_value=0)
    scenario_eli.plot(kind="bar", stacked=True, ax=ax,
                      color=[PALETTE.get(c, "#999") for c in scenario_eli.columns])
    ax.set_title("Eligibility Distribution by EMI Scenario", fontsize=14)
    ax.set_xlabel("EMI Scenario")
    ax.set_ylabel("Count")
    ax.legend(title="Eligibility")
    plt.xticks(rotation=20)
    plt.tight_layout()
    return _save(fig, "02_scenario_distribution.png")


def plot_financial_distributions(df: pd.DataFrame):
    cols = ["monthly_salary", "credit_score", "bank_balance",
            "requested_amount", "max_monthly_emi"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        for label, color in PALETTE.items():
            subset = df[df["emi_eligibility"] == label][col]
            axes[i].hist(subset, bins=40, alpha=0.55, label=label, color=color)
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].legend(fontsize=8)
    axes[-1].axis("off")
    plt.suptitle("Financial Variable Distributions by Eligibility", fontsize=16, y=1.01)
    plt.tight_layout()
    return _save(fig, "03_financial_distributions.png")


# ─────────────────────────────────────────────
# 3. Correlation analysis
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=["emi_eligibility_enc"], errors="ignore")
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdYlGn",
                center=0, linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    return _save(fig, "04_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 4. Feature relationship plots
# ─────────────────────────────────────────────

def plot_credit_score_vs_eligibility(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in PALETTE.items():
        subset = df[df["emi_eligibility"] == label]["credit_score"]
        ax.hist(subset, bins=50, alpha=0.6, label=label, color=color)
    ax.axvline(700, color="black", linestyle="--", linewidth=1.5, label="700 threshold")
    ax.set_title("Credit Score Distribution by Eligibility", fontsize=14)
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    return _save(fig, "05_credit_score_eligibility.png")


def plot_salary_vs_max_emi(df: pd.DataFrame):
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[e] for e in sample["emi_eligibility"]]
    ax.scatter(sample["monthly_salary"], sample["max_monthly_emi"],
               c=colors, alpha=0.35, s=10)
    patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
    ax.legend(handles=patches)
    ax.set_title("Monthly Salary vs Max Monthly EMI", fontsize=14)
    ax.set_xlabel("Monthly Salary (INR)")
    ax.set_ylabel("Max Monthly EMI (INR)")
    plt.tight_layout()
    return _save(fig, "06_salary_vs_max_emi.png")


def plot_dti_distribution(df: pd.DataFrame):
    if "debt_to_income_ratio" not in df.columns:
        total_exp = (df[["monthly_rent", "school_fees", "college_fees",
                          "travel_expenses", "groceries_utilities",
                          "other_monthly_expenses", "current_emi_amount"]].sum(axis=1))
        df = df.copy()
        df["debt_to_income_ratio"] = total_exp / df["monthly_salary"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in PALETTE.items():
        subset = df[df["emi_eligibility"] == label]["debt_to_income_ratio"].clip(0, 2)
        ax.hist(subset, bins=50, alpha=0.6, label=label, color=color)
    ax.axvline(0.40, color="black", linestyle="--", linewidth=1.5, label="40% threshold")
    ax.axvline(0.55, color="gray",  linestyle="--", linewidth=1.5, label="55% threshold")
    ax.set_title("Debt-to-Income Ratio by Eligibility", fontsize=14)
    ax.set_xlabel("DTI Ratio")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    return _save(fig, "07_dti_distribution.png")


def plot_demographic_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cats = ["gender", "marital_status", "education", "employment_type"]
    for ax, col in zip(axes.flatten(), cats):
        ct = pd.crosstab(df[col], df["emi_eligibility"], normalize="index") * 100
        ct[["Eligible", "High_Risk", "Not_Eligible"]].plot(
            kind="bar", stacked=True, ax=ax,
            color=[PALETTE["Eligible"], PALETTE["High_Risk"], PALETTE["Not_Eligible"]])
        ax.set_title(f"Eligibility by {col.replace('_',' ').title()}")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    return _save(fig, "08_demographic_analysis.png")


# ─────────────────────────────────────────────
# 5. Target analysis
# ─────────────────────────────────────────────

def plot_max_emi_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["max_monthly_emi"], bins=60, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axvline(df["max_monthly_emi"].mean(), color="red",
               linestyle="--", linewidth=1.5, label=f"Mean: ₹{df['max_monthly_emi'].mean():,.0f}")
    ax.axvline(df["max_monthly_emi"].median(), color="orange",
               linestyle="--", linewidth=1.5, label=f"Median: ₹{df['max_monthly_emi'].median():,.0f}")
    ax.set_title("Max Monthly EMI Distribution (Regression Target)", fontsize=14)
    ax.set_xlabel("Max Monthly EMI (INR)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    return _save(fig, "09_max_emi_distribution.png")


# ─────────────────────────────────────────────
# 6. Run all
# ─────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> list:
    """Generate all EDA plots and return list of saved paths."""
    paths = []
    paths.append(plot_eligibility_distribution(df))
    paths.append(plot_scenario_distribution(df))
    paths.append(plot_financial_distributions(df))
    paths.append(plot_correlation_heatmap(df))
    paths.append(plot_credit_score_vs_eligibility(df))
    paths.append(plot_salary_vs_max_emi(df))
    paths.append(plot_dti_distribution(df))
    paths.append(plot_demographic_analysis(df))
    paths.append(plot_max_emi_distribution(df))
    print(f"EDA complete. {len(paths)} charts saved to '{OUTPUT_DIR}/'")
    return paths


if __name__ == "__main__":
    from data_preprocessing import generate_dataset, clean_data, engineer_features
    print("Loading data …")
    raw = generate_dataset(n_per_scenario=10000)   # use small sample for quick run
    df  = engineer_features(clean_data(raw))
    stats = basic_stats(df)
    print(f"Shape: {stats['shape']}  |  Missing: {stats['missing']}  |  Duplicates: {stats['duplicates']}")
    print(stats["eligibility_dist"])
    run_full_eda(df)