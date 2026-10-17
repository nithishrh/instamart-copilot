"""Part A — Data Foundation: Clean, Store, Query (22 marks)
Audits raw CSV files, applies documented cleaning rules to customers, products, and orders,
saves cleaned CSVs to data/clean/ and SQLite tables to data/instamart.db, validates reference
checkpoints, runs the 7 SQL queries in sql/queries.sql, and performs the pandas cross-check
and exploratory analysis.
"""
import os
import re
import sqlite3
import numpy as np
import pandas as pd


def audit_raw_data():
    print("=== Task 2: Raw Data Audit ===")
    cust_raw = pd.read_csv("data/raw/customers.csv")
    prod_raw = pd.read_csv("data/raw/products.csv")
    ord_raw = pd.read_csv("data/raw/orders.csv")

    for name, df in [("customers.csv", cust_raw), ("products.csv", prod_raw), ("orders.csv", ord_raw)]:
        print(f"\nTable: {name} | Shape: {df.shape}")
        for col in df.columns:
            na_cnt = df[col].isna().sum()
            na_pct = (na_cnt / len(df)) * 100
            print(f"  {col}: dtype={df[col].dtype}, missing={na_cnt} ({na_pct:.2f}%)")

    cust_s = pd.read_csv("data/raw/customers.csv", keep_default_na=False)
    ord_s = pd.read_csv("data/raw/orders.csv", keep_default_na=False)

    ord_rows = len(ord_s)
    ord_exact_dupes = int(ord_s.duplicated().sum())
    ord_pending = int((ord_s["status"] == "pending").sum())
    ord_distinct_id = int(ord_s["order_id"].nunique())
    amt_num = pd.to_numeric(ord_s["order_amount"], errors="coerce")
    ord_non_num_amt = int(amt_num.isna().sum())
    deliv_num = pd.to_numeric(ord_s["delivery_minutes"], errors="coerce")
    ord_neg_deliv = int((deliv_num < 0).sum())
    ord_blank_deliv = int((ord_s["delivery_minutes"] == "").sum())

    cust_blank_age = int((cust_s["age"] == "").sum())
    age_num = pd.to_numeric(cust_s["age"], errors="coerce")
    cust_bad_age = int(((age_num < 10) | (age_num > 100)).sum())
    cust_raw_cities = int(cust_s["city"].nunique())
    cust_raw_genders = int(cust_s.loc[cust_s["gender"] != "", "gender"].nunique())
    cust_blank_gender = int((cust_s["gender"] == "").sum())

    audit_summary = {
        "order_rows": ord_rows,
        "exact_duplicates": ord_exact_dupes,
        "pending_rows": ord_pending,
        "distinct_order_id": ord_distinct_id,
        "non_numeric_amounts": ord_non_num_amt,
        "negative_delivery_minutes": ord_neg_deliv,
        "blank_delivery_minutes": ord_blank_deliv,
        "blank_ages": cust_blank_age,
        "ages_outside_10_100": cust_bad_age,
        "raw_city_spellings": cust_raw_cities,
        "raw_gender_spellings": cust_raw_genders,
        "blank_genders": cust_blank_gender,
    }
    print("\nAudit Summary:", audit_summary)
    return audit_summary


def clean_customers():
    print("\n=== Task 3: Clean customers ===")
    df = pd.read_csv("data/raw/customers.csv", keep_default_na=False)

    # (a) Standardise city: strip spaces and title case
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # (b) Map every gender spelling to male, female, or unknown (blank -> unknown)
    def map_gender(val):
        v = str(val).strip().lower()
        if v in ("male", "m"):
            return "male"
        if v in ("female", "f"):
            return "female"
        return "unknown"

    df["gender"] = df["gender"].apply(map_gender)

    # (c) Treat ages outside 10-100 as missing, then fill with median age of customer's city
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = np.nan
    df["age"] = df["age"].fillna(df.groupby("city")["age"].transform("median"))

    # (d) Convert signup_date to datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    print("Cleaned customers city counts:\n", df["city"].value_counts().to_dict())
    print("Cleaned customers gender counts:\n", df["gender"].value_counts().to_dict())
    print("Missing ages:", df["age"].isna().sum(), "| Min age:", df["age"].min(), "| Max age:", df["age"].max())
    return df


def clean_orders(products_df):
    print("\n=== Task 4: Clean orders ===")
    df = pd.read_csv("data/raw/orders.csv", keep_default_na=False)

    # i. Convert order_ts to a datetime
    df["order_ts"] = pd.to_datetime(df["order_ts"])

    # ii. Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # iii. For each order_id keep only the record with the latest order_ts
    df = df.sort_values("order_ts").groupby("order_id", as_index=False).last()
    df = df.sort_values("order_id").reset_index(drop=True)

    # iv. Convert order_amount to numeric, fill missing from product price:
    # order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2)
    df["quantity"] = pd.to_numeric(df["quantity"])
    df["discount_pct"] = pd.to_numeric(df["discount_pct"])
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")

    prod_prices = products_df.set_index("product_id")["price_inr"]
    prod_cats = products_df.set_index("product_id")["category"]
    calc_amount = (df["product_id"].map(prod_prices) * df["quantity"] * (1.0 - df["discount_pct"] / 100.0)).round(2)
    df["order_amount"] = df["order_amount"].fillna(calc_amount)

    # v. Set negative delivery_minutes to missing. For completed orders, fill missing with median
    # delivery time of that order's product category among completed orders (groupby + transform).
    # For cancelled orders, leave delivery_minutes missing.
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    df["_category"] = df["product_id"].map(prod_cats)
    completed_mask = df["status"] == "completed"
    df.loc[completed_mask, "delivery_minutes"] = df.loc[completed_mask].groupby("_category")["delivery_minutes"].transform(
        lambda s: s.fillna(s.median())
    )
    df.drop(columns=["_category"], inplace=True)

    print("Cleaned orders rows:", len(df))
    print("Status counts:", df["status"].value_counts().to_dict())
    print("Missing order_amount:", df["order_amount"].isna().sum())
    print("Missing delivery_minutes (completed):", df.loc[df["status"] == "completed", "delivery_minutes"].isna().sum())
    print("Missing delivery_minutes (cancelled):", df.loc[df["status"] == "cancelled", "delivery_minutes"].isna().sum())
    print("Total completed order_amount: INR {:.2f}".format(df.loc[df["status"] == "completed", "order_amount"].sum()))
    return df


def store_and_validate(cust_df, prod_df, ord_df):
    print("\n=== Task 5: Store and Validate ===")
    os.makedirs("data/clean", exist_ok=True)

    cust_out = cust_df.copy()
    cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")

    ord_out = ord_df.copy()
    ord_out["order_ts"] = ord_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cust_out.to_csv("data/clean/customers.csv", index=False)
    prod_df.to_csv("data/clean/products.csv", index=False)
    ord_out.to_csv("data/clean/orders.csv", index=False)

    db_path = "data/instamart.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cust_out.to_sql("customers", conn, index=False, if_exists="replace")
    prod_df.to_sql("products", conn, index=False, if_exists="replace")
    ord_out.to_sql("orders", conn, index=False, if_exists="replace")
    conn.close()

    # Validate reference checkpoints from Tasks 3 and 4
    city_counts = cust_df["city"].value_counts().to_dict()
    assert city_counts == {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }, f"City counts mismatch: {city_counts}"

    gender_counts = cust_df["gender"].value_counts().to_dict()
    assert gender_counts == {"female": 610, "male": 560, "unknown": 30}, f"Gender counts mismatch: {gender_counts}"
    assert cust_df["age"].isna().sum() == 0, "Missing ages remain"
    assert (cust_df["age"] >= 18).all() and (cust_df["age"] <= 60).all(), "Age out of 18..60 range"

    assert len(ord_df) == 12309 and ord_df["order_id"].nunique() == 12309, "Order count mismatch"
    status_counts = ord_df["status"].value_counts().to_dict()
    assert status_counts.get("completed") == 11110, "Completed count mismatch"
    assert status_counts.get("cancelled") == 1199, "Cancelled count mismatch"
    assert status_counts.get("pending", 0) == 0, "Pending rows remain"
    assert ord_df["order_amount"].isna().sum() == 0, "Missing order_amount remains"
    assert ord_df.loc[ord_df["status"] == "completed", "delivery_minutes"].isna().sum() == 0, "Missing completed delivery_minutes"
    assert ord_df.loc[ord_df["status"] == "cancelled", "delivery_minutes"].isna().sum() == 1199, "Cancelled delivery_minutes not missing"
    comp_sum = round(ord_df.loc[ord_df["status"] == "completed", "order_amount"].sum(), 2)
    assert comp_sum == 2998668.80, f"Completed order_amount sum mismatch: {comp_sum}"

    print("ALL CHECKPOINTS PASSED")


def run_sql_queries():
    print("\n=== Task 6: SQL Analysis ===")
    with open("sql/queries.sql", "r", encoding="utf-8") as f:
        sql_text = f.read()

    raw_queries = [q.strip() for q in re.split(r"(?=--\s*Q\d+:)", sql_text) if q.strip()]
    conn = sqlite3.connect("data/instamart.db")

    results = {}
    for idx, q in enumerate(raw_queries, 1):
        df_q = pd.read_sql_query(q, conn)
        results[f"Q{idx}"] = df_q
        print(f"\n--- Q{idx} (rows={len(df_q)}) ---")
        print(df_q.head(6))

    assert len(results["Q1"]) == 30
    assert len(results["Q2"]) == 30
    assert round(results["Q2"]["revenue"].sum(), 2) == 2998668.80
    assert results["Q2"].iloc[0]["city"] == "Bengaluru" and results["Q2"].iloc[0]["category"] == "Electronics"
    assert round(results["Q2"].iloc[0]["revenue"], 2) == 233909.45
    assert len(results["Q3"]) == 11
    assert len(results["Q4"]) == 25
    assert len(results["Q5"]) == 550
    assert round(results["Q5"]["avg_total_spend"].iloc[0], 2) == 2552.06
    assert round(results["Q5"]["total_spend"].iloc[0], 2) == 14766.55

    q6 = results["Q6"]
    top_per_city = q6[q6["row_num"] == 1].set_index("city")[["customer_id", "total_spend"]].to_dict("index")
    print("Q6 Top customer per city:", top_per_city)
    diff_rank_row = int((q6["rnk"] != q6["row_num"]).sum())
    print("Q6 rows with RANK() != ROW_NUMBER():", diff_rank_row)
    assert diff_rank_row == 8

    q7 = results["Q7"]
    assert len(q7) == 183
    assert round(q7["cumulative_revenue"].iloc[-1], 2) == 2998668.80

    conn.close()
    return results


def exploratory_analysis(cust_df, prod_df, ord_df):
    print("\n=== Task 7: pandas Cross-Check and Exploratory Analysis ===")
    comp = ord_df[ord_df["status"] == "completed"].merge(cust_df, on="customer_id").merge(prod_df, on="product_id")

    piv = pd.pivot_table(
        comp,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
    ).round(2)
    print("\nPivot Table (Revenue by City & Category):\n", piv)
    grand_total = round(piv.loc["All", "All"], 2)
    assert grand_total == 2998668.80, f"Pivot grand total mismatch: {grand_total}"

    city_agg = (
        comp.groupby("city")
        .agg(
            sum_amount=("order_amount", "sum"),
            mean_amount=("order_amount", "mean"),
            count_orders=("order_amount", "count"),
        )
        .round(2)
    )
    print("\nCity Aggregation (Completed Orders):\n", city_agg)

    s = comp["order_amount"]
    mean_val = s.mean()
    median_val = s.median()
    skew_val = s.skew()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_below = int((s < lower_fence).sum())
    outliers_above = int((s > upper_fence).sum())

    print(
        f"\nCompleted order_amount stats: mean={mean_val:.2f}, median={median_val:.2f}, "
        f"skew={skew_val:.2f}, Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
        f"lower_fence={lower_fence:.2f} (outliers={outliers_below}), "
        f"upper_fence={upper_fence:.2f} (outliers={outliers_above})"
    )

    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_mat = comp[corr_cols].corr()
    print("\nCorrelation Matrix:\n", corr_mat.round(4))


def main():
    audit_raw_data()
    prod_df = pd.read_csv("data/raw/products.csv")
    prod_df["price_inr"] = pd.to_numeric(prod_df["price_inr"])
    cust_df = clean_customers()
    ord_df = clean_orders(prod_df)
    store_and_validate(cust_df, prod_df, ord_df)
    run_sql_queries()
    exploratory_analysis(cust_df, prod_df, ord_df)


if __name__ == "__main__":
    main()
