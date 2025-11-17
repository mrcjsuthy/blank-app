import pandas as pd
import numpy as np


def fit_elasticity(sku_df, price_col, units_col):
    d = sku_df[[price_col, units_col]].dropna()
    d = d[(d[price_col] > 0) & (d[units_col] > 0)]
    if d.shape[0] < 4:
        return None
    x = np.log(d[price_col].values)
    y = np.log(d[units_col].values)
    A = np.vstack([np.ones_like(x), x]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        intercept, slope = coef[0], coef[1]
        return float(slope)
    except Exception:
        return None


def evaluate_scenarios(current_price, current_units, elasticity, current_margin_per_unit, pct_changes=[-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]):
    if pd.isna(current_margin_per_unit):
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
        rows.append((p, new_price, pred_units, new_margin_per_unit, total_margin))
    return rows


def analyze(df):
    results = []
    for sku, group in df.groupby('sku'):
        group = group.sort_values('date')
        current_price = float(group['price'].iloc[-1])
        current_units = float(group['units'].iloc[-1])
        current_margin = float(group['margin_per_unit'].iloc[-1]) if 'margin_per_unit' in group.columns else None
        elasticity = fit_elasticity(group, 'price', 'units')
        scenarios = evaluate_scenarios(current_price, current_units, elasticity, current_margin)
        best = max(scenarios, key=lambda r: r[4])
        current_total_margin = current_units * (current_price - (current_price - (current_margin if pd.notna(current_margin) else 0.3 * current_price)))
        uplift = best[4] - current_total_margin
        results.append({
            'sku': sku,
            'elasticity': elasticity,
            'current_price': current_price,
            'best_price': best[1],
            'best_pct_change': best[0],
            'pred_units': best[2],
            'pred_total_margin': best[4],
            'current_total_margin': current_total_margin,
            'uplift': uplift,
        })
    res_df = pd.DataFrame(results).sort_values('uplift', ascending=False)
    return res_df


def main():
    df = pd.read_csv('sample_data.csv', parse_dates=['date'])
    res = analyze(df)
    print('Top recommendations (smoke test):')
    print(res.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
