
import os, sys, warnings
import joblib, numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_PATH  = os.path.join(BASE_DIR, "emi_dataset.csv")

# ─────────────────────────────────────────────
# Load models (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        clf    = joblib.load(os.path.join(MODELS_DIR, "best_clf_model.pkl"))
        reg    = joblib.load(os.path.join(MODELS_DIR, "best_reg_model.pkl"))
        enc    = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        return clf, reg, enc, scaler, True
    except Exception as e:
        return None, None, None, None, False


@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


clf_model, reg_model, encoders, scaler, models_ready = load_models()
df_main = load_data()

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:#f0f4ff;border-radius:12px;padding:16px 20px;margin:6px 0;
             border-left:4px solid #4361ee;}
.eligible{color:#2ecc71;font-weight:700;font-size:1.4rem;}
.high_risk{color:#f39c12;font-weight:700;font-size:1.4rem;}
.not_eligible{color:#e74c3c;font-weight:700;font-size:1.4rem;}
.section-header{font-size:1.3rem;font-weight:600;color:#2d3436;margin-top:1.5rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────
PAGES = {
    "🏠 Home":                "home",
    "🔍 Eligibility Check":   "clf",
    "💰 Max EMI Estimator":   "reg",
    "📊 EDA Dashboard":       "eda",
    "🏆 Model Performance":   "perf",
    "🧪 MLflow Tracker":      "mlflow",
    "🗄️ Data Management":     "crud",
}

st.sidebar.image("https://img.icons8.com/fluency/96/000000/bank-cards.png", width=80)
st.sidebar.title("EMIPredict AI")
st.sidebar.markdown("*Intelligent Financial Risk Assessment*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
current = PAGES[page]

if not models_ready:
    st.sidebar.warning("⚠️ Models not found. Run `train_models.py` first.")

st.sidebar.markdown("---")
st.sidebar.info("**Dataset**: 400K records\n\n**Models**: Classification + Regression\n\n**MLflow**: Experiment tracking")

# ═══════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════
if current == "home":
    st.title("💳 EMIPredict AI")
    st.subheader("Intelligent Financial Risk Assessment Platform")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", "400,000", "5 scenarios")
    with c2:
        st.metric("Input Features", "22", "+ 12 engineered")
    with c3:
        st.metric("ML Models", "8", "4 clf + 4 reg")
    with c4:
        st.metric("Target Variables", "2", "Clf + Reg")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🎯 What This Platform Does")
        st.markdown("""
- **EMI Eligibility Prediction** — classify applicants as *Eligible*, *High Risk*, or *Not Eligible*
- **Max EMI Estimation** — predict the maximum safe monthly EMI an applicant can afford
- **Interactive EDA** — explore 400K financial profiles with dynamic charts
- **Model Comparison** — compare all trained models via MLflow tracking
- **Data CRUD** — add, view, edit, and delete applicant records
""")
    with col_b:
        st.markdown("### 📦 EMI Scenarios")
        scenario_data = {
            "Scenario": ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"],
            "Amount Range (INR)": ["10K – 2L", "20K – 3L", "80K – 15L", "50K – 10L", "50K – 5L"],
            "Tenure (months)":    ["3 – 24", "6 – 36", "12 – 84", "12 – 60", "6 – 48"],
        }
        st.dataframe(pd.DataFrame(scenario_data), use_container_width=True, hide_index=True)

    if df_main is not None:
        st.markdown("---")
        st.markdown("### 📈 Quick Dataset Overview")
        q1, q2, q3 = st.columns(3)
        with q1:
            fig = px.pie(df_main, names="emi_eligibility", title="Eligibility Distribution",
                         color="emi_eligibility",
                         color_discrete_map={"Eligible":"#2ecc71","High_Risk":"#f39c12","Not_Eligible":"#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)
        with q2:
            fig = px.histogram(df_main, x="credit_score", color="emi_eligibility",
                               barmode="overlay", nbins=50,
                               title="Credit Score Distribution",
                               color_discrete_map={"Eligible":"#2ecc71","High_Risk":"#f39c12","Not_Eligible":"#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)
        with q3:
            grp = df_main.groupby("emi_scenario")["emi_eligibility"].value_counts(normalize=True).unstack() * 100
            fig = px.bar(grp.reset_index(), x="emi_scenario",
                         y=["Eligible","High_Risk","Not_Eligible"],
                         title="Eligibility % by Scenario", barmode="stack",
                         color_discrete_map={"Eligible":"#2ecc71","High_Risk":"#f39c12","Not_Eligible":"#e74c3c"})
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2 — CLASSIFICATION
# ═══════════════════════════════════════════════
elif current == "clf":
    st.title("🔍 EMI Eligibility Check")
    st.markdown("Fill in the applicant's financial details to get an eligibility prediction.")

    if not models_ready:
        st.error("Models not loaded. Please run `train_models.py` first.")
        st.stop()

    # ── Input form ──
    with st.form("clf_form"):
        st.markdown("#### Personal & Employment Details")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        age               = r1c1.number_input("Age", 25, 60, 35)
        gender            = r1c2.selectbox("Gender", ["Male", "Female"])
        marital_status    = r1c3.selectbox("Marital Status", ["Single", "Married"])
        education         = r1c4.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        employment_type      = r2c1.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_of_employment  = r2c2.number_input("Years Employed", 1, 35, 5)
        company_type         = r2c3.selectbox("Company Type", ["Large", "Medium", "Small", "Startup"])
        house_type           = r2c4.selectbox("House Type", ["Rented", "Own", "Family"])

        st.markdown("#### Financial Details")
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        monthly_salary    = r3c1.number_input("Monthly Salary (₹)", 15000, 300000, 50000, step=1000)
        monthly_rent      = r3c2.number_input("Monthly Rent (₹)", 0, 50000, 10000, step=500)
        credit_score      = r3c3.number_input("Credit Score", 300, 850, 650)
        bank_balance      = r3c4.number_input("Bank Balance (₹)", 0, 2000000, 100000, step=5000)

        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        family_size       = r4c1.number_input("Family Size", 1, 10, 3)
        dependents        = r4c2.number_input("Dependents", 0, 8, 1)
        emergency_fund    = r4c3.number_input("Emergency Fund (₹)", 0, 500000, 50000, step=5000)
        existing_loans    = r4c4.selectbox("Existing Loans", ["No", "Yes"])

        st.markdown("#### Monthly Expenses")
        r5c1, r5c2, r5c3, r5c4, r5c5 = st.columns(5)
        school_fees       = r5c1.number_input("School Fees (₹)", 0, 20000, 2000, step=500)
        college_fees      = r5c2.number_input("College Fees (₹)", 0, 30000, 0, step=500)
        travel_expenses   = r5c3.number_input("Travel (₹)", 500, 20000, 3000, step=500)
        groceries_util    = r5c4.number_input("Groceries & Util (₹)", 1000, 50000, 8000, step=500)
        other_expenses    = r5c5.number_input("Other Expenses (₹)", 500, 30000, 3000, step=500)
        current_emi       = st.number_input("Current EMI Amount (₹)", 0, 50000, 0, step=500) if existing_loans == "Yes" else 0

        st.markdown("#### Loan Request")
        lc1, lc2, lc3 = st.columns(3)
        emi_scenario      = lc1.selectbox("EMI Scenario", list({"E-commerce":0,"Home_Appliances":0,"Vehicle":0,"Personal_Loan":0,"Education":0}.keys()))
        requested_amount  = lc2.number_input("Requested Amount (₹)", 10000, 1500000, 200000, step=10000)
        requested_tenure  = lc3.number_input("Requested Tenure (months)", 3, 84, 24)

        submitted = st.form_submit_button("🔮 Predict Eligibility", type="primary", use_container_width=True)

    if submitted:
        from data_preprocessing import engineer_features, CATEGORICAL_COLS
        input_dict = {
            "age": age, "gender": gender, "marital_status": marital_status,
            "education": education, "monthly_salary": monthly_salary,
            "employment_type": employment_type, "years_of_employment": years_of_employment,
            "company_type": company_type, "house_type": house_type,
            "monthly_rent": monthly_rent, "family_size": family_size,
            "dependents": dependents, "school_fees": school_fees,
            "college_fees": college_fees, "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_util, "other_monthly_expenses": other_expenses,
            "existing_loans": existing_loans, "current_emi_amount": current_emi,
            "credit_score": credit_score, "bank_balance": bank_balance,
            "emergency_fund": emergency_fund, "emi_scenario": emi_scenario,
            "requested_amount": requested_amount, "requested_tenure": requested_tenure,
        }
        input_df = pd.DataFrame([input_dict])
        input_df = engineer_features(input_df)

        # encode
        for col in CATEGORICAL_COLS:
            if col in input_df.columns and col in encoders:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))

        num_feat = [c for c in input_df.columns if c in scaler.feature_names_in_]
        input_df[num_feat] = scaler.transform(input_df[num_feat])

        X_pred = input_df[[c for c in clf_model.feature_names_in_ if c in input_df.columns]]
        prediction = clf_model.predict(X_pred)[0]
        label_map  = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
        result     = label_map.get(prediction, str(prediction))

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        col_res, col_detail = st.columns([1, 2])
        with col_res:
            css_cls = result.lower()
            icons   = {"Eligible": "✅", "High_Risk": "⚠️", "Not_Eligible": "❌"}
            st.markdown(f"<div class='metric-card' style='text-align:center'>"
                        f"<div style='font-size:3rem'>{icons.get(result,'🔮')}</div>"
                        f"<div class='{css_cls}'>{result.replace('_',' ')}</div>"
                        f"</div>", unsafe_allow_html=True)
            if hasattr(clf_model, "predict_proba"):
                proba = clf_model.predict_proba(X_pred)[0]
                st.markdown("**Confidence scores**")
                for cls_idx, cls_name in label_map.items():
                    st.progress(float(proba[cls_idx]), text=f"{cls_name}: {proba[cls_idx]*100:.1f}%")

        with col_detail:
            total_exp = (monthly_rent + school_fees + college_fees + travel_expenses
                         + groceries_util + other_expenses + current_emi)
            disposable = monthly_salary - total_exp
            dti = total_exp / max(monthly_salary, 1)
            st.markdown("**Financial Summary**")
            d = {
                "Monthly Salary": f"₹{monthly_salary:,}",
                "Total Expenses":  f"₹{total_exp:,}",
                "Disposable Income": f"₹{disposable:,}",
                "Debt-to-Income Ratio": f"{dti*100:.1f}%",
                "Credit Score": credit_score,
                "Bank Balance": f"₹{bank_balance:,}",
            }
            for k, v in d.items():
                st.markdown(f"- **{k}**: {v}")

            recs = {
                "Eligible": "✅ The applicant demonstrates strong financial health. Loan approval recommended.",
                "High_Risk": "⚠️ Marginal case. Consider higher interest rates or reduced loan amount.",
                "Not_Eligible": "❌ High default risk detected. Loan not recommended at current parameters.",
            }
            st.info(recs[result])


# ═══════════════════════════════════════════════
# PAGE 3 — REGRESSION
# ═══════════════════════════════════════════════
elif current == "reg":
    st.title("💰 Max Monthly EMI Estimator")
    st.markdown("Predict the maximum safe EMI amount an applicant can afford each month.")

    if not models_ready:
        st.error("Models not loaded. Please run `train_models.py` first.")
        st.stop()

    with st.form("reg_form"):
        c1, c2, c3 = st.columns(3)
        monthly_salary   = c1.number_input("Monthly Salary (₹)", 15000, 300000, 60000, step=1000)
        credit_score     = c2.number_input("Credit Score", 300, 850, 700)
        bank_balance     = c3.number_input("Bank Balance (₹)", 0, 2000000, 150000, step=5000)

        c4, c5, c6 = st.columns(3)
        monthly_rent     = c4.number_input("Monthly Rent (₹)", 0, 50000, 8000, step=500)
        travel_expenses  = c5.number_input("Travel Expenses (₹)", 500, 20000, 3000, step=500)
        groceries_util   = c6.number_input("Groceries & Utilities (₹)", 1000, 50000, 8000, step=500)

        c7, c8, c9 = st.columns(3)
        school_fees      = c7.number_input("School Fees (₹)", 0, 20000, 1500, step=500)
        other_expenses   = c8.number_input("Other Expenses (₹)", 500, 30000, 2000, step=500)
        current_emi      = c9.number_input("Current EMI (₹)", 0, 50000, 0, step=500)

        c10, c11, c12, c13 = st.columns(4)
        age              = c10.number_input("Age", 25, 60, 35)
        years_emp        = c11.number_input("Years Employed", 1, 35, 7)
        family_size      = c12.number_input("Family Size", 1, 10, 3)
        dependents       = c13.number_input("Dependents", 0, 8, 1)

        c14, c15, c16, c17 = st.columns(4)
        gender           = c14.selectbox("Gender", ["Male", "Female"])
        marital_status   = c15.selectbox("Marital Status", ["Single", "Married"])
        education        = c16.selectbox("Education", ["High School","Graduate","Post Graduate","Professional"])
        employment_type  = c17.selectbox("Employment Type", ["Private","Government","Self-employed"])

        c18, c19, c20 = st.columns(3)
        company_type     = c18.selectbox("Company Type", ["Large","Medium","Small","Startup"])
        house_type       = c19.selectbox("House Type", ["Rented","Own","Family"])
        existing_loans   = c20.selectbox("Existing Loans", ["No","Yes"])

        c21, c22, c23 = st.columns(3)
        emergency_fund   = c21.number_input("Emergency Fund (₹)", 0, 500000, 60000, step=5000)
        emi_scenario     = c22.selectbox("EMI Scenario", ["E-commerce","Home_Appliances","Vehicle","Personal_Loan","Education"])
        requested_amount = c23.number_input("Loan Amount (₹)", 10000, 1500000, 300000, step=10000)
        requested_tenure = st.number_input("Tenure (months)", 3, 84, 36)
        college_fees     = 0

        submitted_reg = st.form_submit_button("💰 Estimate Max EMI", type="primary", use_container_width=True)

    if submitted_reg:
        from data_preprocessing import engineer_features, CATEGORICAL_COLS

        input_dict = {
            "age": age, "gender": gender, "marital_status": marital_status,
            "education": education, "monthly_salary": monthly_salary,
            "employment_type": employment_type, "years_of_employment": years_emp,
            "company_type": company_type, "house_type": house_type,
            "monthly_rent": monthly_rent, "family_size": family_size,
            "dependents": dependents, "school_fees": school_fees,
            "college_fees": college_fees, "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_util, "other_monthly_expenses": other_expenses,
            "existing_loans": existing_loans, "current_emi_amount": current_emi,
            "credit_score": credit_score, "bank_balance": bank_balance,
            "emergency_fund": emergency_fund, "emi_scenario": emi_scenario,
            "requested_amount": requested_amount, "requested_tenure": requested_tenure,
        }
        input_df = pd.DataFrame([input_dict])
        input_df = engineer_features(input_df)
        for col in CATEGORICAL_COLS:
            if col in input_df.columns and col in encoders:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
        num_feat = [c for c in input_df.columns if c in scaler.feature_names_in_]
        input_df[num_feat] = scaler.transform(input_df[num_feat])
        X_pred = input_df[[c for c in reg_model.feature_names_in_ if c in input_df.columns]]
        pred_emi = reg_model.predict(X_pred)[0]
        pred_emi = max(500, min(50000, pred_emi))

        st.markdown("---")
        st.markdown("### 💰 Prediction Result")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Max Monthly EMI", f"₹{pred_emi:,.0f}")
        total_exp = monthly_rent + school_fees + travel_expenses + groceries_util + other_expenses + current_emi
        rc2.metric("Total Monthly Expenses", f"₹{total_exp:,}")
        rc3.metric("Disposable Income", f"₹{monthly_salary - total_exp:,}")

        # EMI gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_emi,
            title={"text": "Max Safe EMI (₹)"},
            gauge={
                "axis": {"range": [0, 50000]},
                "bar": {"color": "#4361ee"},
                "steps": [
                    {"range": [0, 10000], "color": "#e74c3c"},
                    {"range": [10000, 25000], "color": "#f39c12"},
                    {"range": [25000, 50000], "color": "#2ecc71"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4 — EDA DASHBOARD
# ═══════════════════════════════════════════════
elif current == "eda":
    st.title("📊 EDA Dashboard")
    if df_main is None:
        st.warning("Dataset not found. Run `data_preprocessing.py` to generate it.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Correlations", "Demographics"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(df_main):,}")
        c2.metric("Features", len(df_main.columns))
        c3.metric("Missing Values", df_main.isnull().sum().sum())
        c4.metric("Duplicates", df_main.duplicated().sum())
        st.dataframe(df_main.describe().round(2), use_container_width=True)

    with tab2:
        col = st.selectbox("Select variable", ["monthly_salary", "credit_score", "bank_balance",
                                                "max_monthly_emi", "requested_amount", "age"])
        fig = px.histogram(df_main, x=col, color="emi_eligibility", barmode="overlay", nbins=60,
                           color_discrete_map={"Eligible":"#2ecc71","High_Risk":"#f39c12","Not_Eligible":"#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df_main, x="emi_eligibility", y=col, color="emi_eligibility",
                      color_discrete_map={"Eligible":"#2ecc71","High_Risk":"#f39c12","Not_Eligible":"#e74c3c"})
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        num_cols = df_main.select_dtypes(include=[np.number]).columns.tolist()
        selected = st.multiselect("Select features for correlation", num_cols,
                                  default=["monthly_salary","credit_score","bank_balance","max_monthly_emi","age"])
        if selected:
            corr = df_main[selected].corr()
            fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                             title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        cat = st.selectbox("Demographic feature", ["gender","marital_status","education","employment_type","house_type","emi_scenario"])
        ct  = pd.crosstab(df_main[cat], df_main["emi_eligibility"], normalize="index") * 100
        fig = px.bar(ct.reset_index(), x=cat,
                     y=["Eligible","High_Risk","Not_Eligible"] if "Eligible" in ct.columns else ct.columns.tolist(),
                     barmode="stack", title=f"Eligibility % by {cat.replace('_',' ').title()}",
                     color_discrete_map={"Eligible":"#2ecc71","High_Risk":"#f39c12","Not_Eligible":"#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif current == "perf":
    st.title("🏆 Model Performance Comparison")
    st.info("These results are populated after running `train_models.py`.")

    st.markdown("#### Classification Models")
    clf_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Gradient Boosting"],
        "Val Accuracy": ["—"] * 4,
        "Val F1":       ["—"] * 4,
        "Val ROC-AUC":  ["—"] * 4,
    }
    results_path = os.path.join(MODELS_DIR, "results.pkl")
    if os.path.exists(results_path):
        res = joblib.load(results_path)
        for i, name in enumerate(clf_data["Model"]):
            key = name.replace(" ", "_")
            if key in res.get("clf", {}):
                m = res["clf"][key]["metrics"]
                clf_data["Val Accuracy"][i] = f"{m.get('val_accuracy', 0):.4f}"
                clf_data["Val F1"][i]        = f"{m.get('val_f1', 0):.4f}"
                clf_data["Val ROC-AUC"][i]   = f"{m.get('val_roc_auc', 0):.4f}"
    st.dataframe(pd.DataFrame(clf_data), use_container_width=True, hide_index=True)

    st.markdown("#### Regression Models")
    reg_data = {
        "Model": ["Linear Regression", "Random Forest", "XGBoost", "Gradient Boosting"],
        "Val RMSE": ["—"] * 4,
        "Val MAE":  ["—"] * 4,
        "Val R²":   ["—"] * 4,
    }
    st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Classification Metrics Guide")
    st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| Accuracy | Overall correct predictions |
| F1 Score | Harmonic mean of precision and recall (handles class imbalance) |
| ROC-AUC | Area under the ROC curve — discrimination ability |
""")
    st.markdown("#### Regression Metrics Guide")
    st.markdown("""
| Metric | What it measures |
|--------|-----------------|
| RMSE | Root Mean Squared Error — sensitive to large errors |
| MAE  | Mean Absolute Error — average prediction error (₹) |
| R²   | Proportion of variance explained (1.0 = perfect) |
""")


# ═══════════════════════════════════════════════
# PAGE 6 — MLFLOW TRACKER
# ═══════════════════════════════════════════════
elif current == "mlflow":
    st.title("🧪 MLflow Experiment Tracker")
    st.markdown("""
MLflow tracks every training run — parameters, metrics, artifacts, and models.

**To view the full MLflow UI locally:**
```bash
mlflow ui --port 5000
```
Then open [http://localhost:5000](http://localhost:5000)

**Experiment names:**
- `EMIPredict_Classification`
- `EMIPredict_Regression`
""")
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        exps = client.search_experiments()
        if exps:
            st.markdown("### Tracked Experiments")
            for exp in exps:
                with st.expander(f"📁 {exp.name}"):
                    runs = client.search_runs(experiment_ids=[exp.experiment_id],
                                              order_by=["metrics.val_f1 DESC"])
                    if runs:
                        rows = []
                        for r in runs:
                            row = {"Run": r.info.run_name or r.info.run_id[:8]}
                            row.update({k: round(v, 4) for k, v in r.data.metrics.items()})
                            rows.append(row)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.write("No runs yet.")
        else:
            st.warning("No MLflow experiments found. Run `train_models.py` first.")
    except Exception as e:
        st.warning(f"MLflow not available: {e}")


# ═══════════════════════════════════════════════
# PAGE 7 — DATA MANAGEMENT (CRUD)
# ═══════════════════════════════════════════════
elif current == "crud":
    st.title("🗄️ Data Management")

    if "crud_df" not in st.session_state:
        if df_main is not None:
            st.session_state.crud_df = df_main.copy()
        else:
            st.session_state.crud_df = pd.DataFrame()

    crud_df = st.session_state.crud_df
    tab_view, tab_add, tab_edit, tab_delete = st.tabs(["📋 View", "➕ Add", "✏️ Edit", "🗑️ Delete"])

    with tab_view:
        st.metric("Total Records", f"{len(crud_df):,}")
        filters = st.expander("🔍 Filter data")
        with filters:
            fc1, fc2 = st.columns(2)
            eli_filter = fc1.multiselect("Eligibility", ["Eligible","High_Risk","Not_Eligible"],
                                         default=["Eligible","High_Risk","Not_Eligible"])
            scen_filter = fc2.multiselect("Scenario", list({"E-commerce":0,"Home_Appliances":0,"Vehicle":0,"Personal_Loan":0,"Education":0}),
                                          default=list({"E-commerce":0,"Home_Appliances":0,"Vehicle":0,"Personal_Loan":0,"Education":0}))
        if len(crud_df):
            filtered = crud_df[
                crud_df["emi_eligibility"].isin(eli_filter) &
                crud_df["emi_scenario"].isin(scen_filter)
            ]
            st.dataframe(filtered.head(500), use_container_width=True)
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "emi_filtered.csv", "text/csv")
        else:
            st.info("No data loaded.")

    with tab_add:
        st.markdown("Add a new applicant record manually.")
        with st.form("add_form"):
            na1, na2, na3 = st.columns(3)
            new_salary   = na1.number_input("Monthly Salary", 15000, 300000, 50000)
            new_credit   = na2.number_input("Credit Score", 300, 850, 650)
            new_scenario = na3.selectbox("Scenario", ["E-commerce","Home_Appliances","Vehicle","Personal_Loan","Education"])
            new_amount   = st.number_input("Requested Amount", 10000, 1500000, 100000)
            new_eli      = st.selectbox("Eligibility (manual label)", ["Eligible","High_Risk","Not_Eligible"])
            add_btn      = st.form_submit_button("Add Record", type="primary")
        if add_btn:
            new_row = {c: np.nan for c in (crud_df.columns if len(crud_df) else [])}
            new_row.update({
                "monthly_salary": new_salary, "credit_score": new_credit,
                "emi_scenario": new_scenario, "requested_amount": new_amount,
                "emi_eligibility": new_eli,
            })
            st.session_state.crud_df = pd.concat(
                [crud_df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Record added!")

    with tab_edit:
        st.markdown("Edit an existing record by row index.")
        if len(crud_df):
            idx = st.number_input("Row index to edit", 0, len(crud_df) - 1, 0)
            row = crud_df.iloc[idx]
            with st.form("edit_form"):
                new_salary_e = st.number_input("Monthly Salary", value=int(row.get("monthly_salary", 50000)))
                new_credit_e = st.number_input("Credit Score",   value=int(row.get("credit_score", 650)))
                new_eli_e    = st.selectbox("Eligibility", ["Eligible","High_Risk","Not_Eligible"],
                                            index=["Eligible","High_Risk","Not_Eligible"].index(
                                                row.get("emi_eligibility","Eligible")) if row.get("emi_eligibility") in ["Eligible","High_Risk","Not_Eligible"] else 0)
                edit_btn = st.form_submit_button("Save Changes", type="primary")
            if edit_btn:
                st.session_state.crud_df.at[idx, "monthly_salary"] = new_salary_e
                st.session_state.crud_df.at[idx, "credit_score"]   = new_credit_e
                st.session_state.crud_df.at[idx, "emi_eligibility"] = new_eli_e
                st.success(f"Row {idx} updated!")
        else:
            st.info("No records to edit.")

    with tab_delete:
        st.warning("⚠️ Deletion is permanent for this session.")
        if len(crud_df):
            del_idx = st.number_input("Row index to delete", 0, len(crud_df) - 1, 0)
            if st.button("🗑️ Delete Row", type="primary"):
                st.session_state.crud_df = crud_df.drop(index=del_idx).reset_index(drop=True)
                st.success(f"Row {del_idx} deleted!")
        else:
            st.info("No records to delete.")