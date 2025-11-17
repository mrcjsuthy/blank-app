import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Price + Display POC", layout="wide")

st.title("🎈 Price & Display - POC")
st.write("Upload historical sales data (CSV or Excel). The app will estimate price elasticity per SKU and recommend price moves to maximize total margin $.")


@st.cache_data
def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        # Excel
        return pd.read_excel(uploaded_file)


def guess_columns(df):
    cols = {c.lower(): c for c in df.columns}
    def find(keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return None
    return {
        "sku": find(["sku", "article", "articlecode", "article_id", "product", "sku_id"]),
        "date": find(["date", "day", "sales_date"]),
        "units": find(["units", "qty", "quantity", "units_sold", "sales_qty"]),
        "price": find(["price", "unit_price", "sale_price", "selling_price", "retail_price"]),
        "margin": find(["margin", "margin_per_unit", "unit_margin", "profit_per_unit"]),
        "on_display": find(["display", "on_display", "is_display", "on_promo"]),
        "category": find(["category", "cat", "department"]),
    }


def fit_elasticity(sku_df, price_col, units_col):
    # require positive values
    d = sku_df[[price_col, units_col]].dropna()
    d = d[(d[price_col] > 0) & (d[units_col] > 0)]
    if d.shape[0] < 4:
        return None
    # log-log regression: log(units) = a + b * log(price)
    x = np.log(d[price_col].values)
    y = np.log(d[units_col].values)
    A = np.vstack([np.ones_like(x), x]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        intercept, slope = coef[0], coef[1]
        elasticity = slope
        return float(elasticity)
    except Exception:
        return None


def evaluate_scenarios(current_price, current_units, elasticity, current_margin_per_unit, pct_changes=[-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]):
    # infer cost from current price and margin per unit
    if pd.isna(current_margin_per_unit):
        # If margin missing, assume margin is 30% of price as fallback
        current_margin_per_unit = 0.30 * current_price
    cost = current_price - current_margin_per_unit
    rows = []
    for p in pct_changes:
        new_price = current_price * (1 + p)
        if new_price <= 0:
            continue
        if elasticity is None:
            pred_units = current_units
        else:
            pred_units = current_units * (new_price / current_price) ** elasticity
        pred_units = max(pred_units, 0)
        new_margin_per_unit = new_price - cost
        total_margin = pred_units * new_margin_per_unit
        rows.append({
            "pct_change": p,
            "new_price": new_price,
            "pred_units": pred_units,
            "new_margin_per_unit": new_margin_per_unit,
            "total_margin": total_margin,
        })
    return pd.DataFrame(rows)


def analyze(df, mapping, top_n=10):
    sku_col = mapping["sku"]
    price_col = mapping["price"]
    units_col = mapping["units"]
    margin_col = mapping.get("margin")

    results = []
    for sku, group in df.groupby(sku_col):
        group = group.sort_values(by=mapping.get("date") or price_col)
        current_price = float(group[price_col].iloc[-1])
        current_units = float(group[units_col].iloc[-1])
        current_margin = None
        if margin_col and margin_col in group.columns:
            current_margin = float(group[margin_col].iloc[-1])
        elasticity = fit_elasticity(group, price_col, units_col)
        scenarios = evaluate_scenarios(current_price, current_units, elasticity, current_margin)
        # choose optimal scenario
        best = scenarios.loc[scenarios["total_margin"].idxmax()]
        current_total_margin = current_units * (current_price - (current_price - (current_margin if pd.notna(current_margin) else 0.3 * current_price)))
        uplift = best["total_margin"] - current_total_margin
        results.append({
            "sku": sku,
            "elasticity": elasticity,
            "current_price": current_price,
            "current_units": current_units,
            "best_price": best["new_price"],
            "best_pct_change": best["pct_change"],
            "pred_units_at_best": best["pred_units"],
            "pred_total_margin": best["total_margin"],
            "current_total_margin": current_total_margin,
            "uplift": uplift,
        })
    res_df = pd.DataFrame(results).sort_values("uplift", ascending=False)
    return res_df


st.sidebar.header("Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=False)

if uploaded is not None:
    df = read_file(uploaded)
    st.sidebar.success("File loaded: {} rows".format(len(df)))

    guessed = guess_columns(df)
    st.header("Column mapping")
    col1, col2 = st.columns([2, 3])
    with col1:
        sku_col = st.selectbox("SKU column", options=[None] + list(df.columns), index=(1 if guessed["sku"] in df.columns else 0))
        price_col = st.selectbox("Price column", options=[None] + list(df.columns), index=(1 if guessed["price"] in df.columns else 0))
        units_col = st.selectbox("Units column", options=[None] + list(df.columns), index=(1 if guessed["units"] in df.columns else 0))
    with col2:
        margin_col = st.selectbox("Margin per unit column (optional)", options=[None] + list(df.columns), index=(1 if guessed.get("margin") in df.columns else 0))
        date_col = st.selectbox("Date column (optional)", options=[None] + list(df.columns), index=(1 if guessed.get("date") in df.columns else 0))

    mapping = {"sku": sku_col, "price": price_col, "units": units_col, "margin": margin_col, "date": date_col}
    if not sku_col or not price_col or not units_col:
        st.warning("Please map SKU, Price and Units columns to continue.")
    else:
        st.subheader("Data preview")
        st.dataframe(df.head(200))

        if st.button("Run analysis"):
            with st.spinner("Estimating elasticity and evaluating scenarios..."):
                res = analyze(df, mapping, top_n=10)
            st.success("Analysis complete")

            st.subheader("Top SKUs by potential margin uplift")
            st.write("Top 10 SKUs the POC suggests moving price for to increase total margin $ (simple log-log elasticity).")
            top10 = res.head(10)
            st.dataframe(top10[["sku","elasticity","current_price","best_price","best_pct_change","current_total_margin","pred_total_margin","uplift"]])

            st.subheader("Top 5 recommended moves (summary)")
            top5 = top10.head(5)
            for _, r in top5.iterrows():
                reason = []
                if pd.notna(r.elasticity):
                    reason.append(f"Estimated elasticity={r.elasticity:.2f}")
                else:
                    reason.append("Elasticity insufficient (fallback), consider experiment")
                reason.append(f"Predicted uplift ${r.uplift:,.2f}")
                st.markdown(f"**{r.sku}** — Move price {r.best_pct_change*100:+.0f}% to ${r.best_price:.2f}. " + " — ".join(reason))

            csv = top10.to_csv(index=False).encode("utf-8")
            st.download_button("Download suggestions CSV", data=csv, file_name="price_suggestions.csv", mime="text/csv")

            st.info("Notes: This is a proof-of-concept model using a simple log-log elasticity estimate per SKU and assumes cost is current_price - current_margin_per_unit. Use as guidance and validate with experiments.")

else:
    st.info("Upload a CSV or Excel file using the sidebar to get started. Expect columns like SKU, date, units, price, margin_per_unit.")

