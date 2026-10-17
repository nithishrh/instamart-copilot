"""Part B — Customer Segments and Lapse Prediction (28 marks)
Builds the RFM table, runs K-Means clustering (K=2..8) and PCA, profiles and saves customer
segments, constructs a leak-free pre-cutoff feature matrix for lapse prediction, tunes
LogisticRegression and RandomForestClassifier via 5-fold GridSearchCV, runs the class-balance
oversampling experiment, persists the selected model, and generates the retention target list.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_rfm(orders_df):
    print("=== Task 1: Build the RFM Table ===")
    comp = orders_df[orders_df["status"] == "completed"].copy()
    comp["order_ts"] = pd.to_datetime(comp["order_ts"])
    snapshot = pd.Timestamp("2025-10-01 00:00:00")

    rfm = (
        comp.groupby("customer_id", as_index=False)
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

    print(f"Customers in RFM: {len(rfm)}")
    print(
        f"Recency: median={rfm['recency'].median():.1f}, min={rfm['recency'].min()}, max={rfm['recency'].max()}"
    )
    print(f"Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
    print(f"Monetary: mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f}")

    assert len(rfm) == 1175
    assert rfm["recency"].median() == 19
    assert rfm["recency"].min() == 0 and rfm["recency"].max() == 180
    assert round(rfm["frequency"].mean(), 2) == 9.46 and rfm["frequency"].max() == 33
    assert round(rfm["monetary"].mean(), 2) == 2552.06 and round(rfm["monetary"].max(), 2) == 14766.55
    return rfm


def cluster_and_profile(rfm):
    print("\n=== Task 2: Cluster with K-Means (K=2..8) ===")
    scaler = StandardScaler()
    X_rfm = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    k_records = []
    models = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_rfm)
        sil = silhouette_score(X_rfm, labels)
        k_records.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(sil, 3)})
        models[k] = km
        print(f"  K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.3f}")

    print("\n=== Task 3: Profile, Name and Project Segments (Chosen K=4) ===")
    chosen_k = 4
    km4 = models[chosen_k]
    rfm_seg = rfm.copy()
    rfm_seg["segment_id"] = km4.labels_

    profile = (
        rfm_seg.groupby("segment_id")
        .agg(
            size=("customer_id", "count"),
            mean_recency=("recency", "mean"),
            mean_frequency=("frequency", "mean"),
            mean_monetary=("monetary", "mean"),
        )
        .round(2)
    )
    print("Cluster Profile (K=4):\n", profile)

    # Identify clusters dynamically by their statistical characteristics
    premium_cluster = int(profile["mean_monetary"].idxmax())
    remaining = [c for c in profile.index if c != premium_cluster]
    power_freq_cluster = int(profile.loc[remaining, "mean_frequency"].idxmax())
    rem2 = [c for c in remaining if c != power_freq_cluster]
    dormant_cluster = int(profile.loc[rem2, "mean_recency"].idxmax())
    steady_cluster = [c for c in rem2 if c != dormant_cluster][0]

    name_map = {
        premium_cluster: "Premium Big-Ticket Buyers",
        power_freq_cluster: "Loyal Frequent Shoppers",
        steady_cluster: "Active Casual Buyers",
        dormant_cluster: "At-Risk Dormant Buyers",
    }

    rfm_seg["segment_name"] = rfm_seg["segment_id"].map(name_map)
    rfm_seg["is_premium"] = rfm_seg["segment_id"] == premium_cluster

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_rfm)
    evr = pca.explained_variance_ratio_
    print(f"PCA explained_variance_ratio_: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, Total={evr.sum():.3f}")

    os.makedirs("outputs", exist_ok=True)
    out_cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    rfm_seg[out_cols].to_csv("outputs/customer_segments.csv", index=False)
    print("Wrote outputs/customer_segments.csv")
    return rfm_seg, profile, evr


def build_lapse_cohort(orders_df, customers_df, products_df):
    print("\n=== Task 4: Define the Lapse Problem (Cutoff 2025-09-01) ===")
    cutoff = pd.Timestamp("2025-09-01 00:00:00")
    ord_df = orders_df.copy()
    ord_df["order_ts"] = pd.to_datetime(ord_df["order_ts"])
    ord_df = ord_df.merge(products_df[["product_id", "category"]], on="product_id", how="left")

    pre_all = ord_df[ord_df["order_ts"] < cutoff].copy()
    pre_comp = pre_all[pre_all["status"] == "completed"].copy()
    post_comp = ord_df[(ord_df["order_ts"] >= cutoff) & (ord_df["status"] == "completed")].copy()

    # Cohort: every customer with at least one completed order before cutoff
    cohort_cids = sorted(pre_comp["customer_id"].unique())
    active_sept_cids = set(post_comp["customer_id"].unique())

    comp_agg = (
        pre_comp.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_delivery_minutes=("delivery_minutes", "mean"),
        )
        .reset_index()
    )
    comp_agg["recency"] = (cutoff - comp_agg["last_order_ts"]).dt.days
    comp_agg.drop(columns=["last_order_ts"], inplace=True)

    # Most frequent product category before cutoff (deterministic tie-break via sorted category name)
    cat_counts = (
        pre_comp.groupby(["customer_id", "category"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["customer_id", "cnt", "category"], ascending=[True, False, True])
        .drop_duplicates("customer_id")
        .rename(columns={"category": "fav_category"})[["customer_id", "fav_category"]]
    )

    # Cancellation rate: cancelled orders / all orders with a final status before cutoff
    status_agg = (
        pre_all.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            cancelled_orders=("status", lambda s: (s == "cancelled").sum()),
        )
        .reset_index()
    )
    status_agg["cancellation_rate"] = status_agg["cancelled_orders"] / status_agg["total_orders"]

    cust = customers_df.copy()
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    cust["tenure"] = (cutoff - cust["signup_date"]).dt.days

    df_cohort = (
        pd.DataFrame({"customer_id": cohort_cids})
        .merge(comp_agg, on="customer_id", how="left")
        .merge(cat_counts, on="customer_id", how="left")
        .merge(status_agg[["customer_id", "cancellation_rate"]], on="customer_id", how="left")
        .merge(cust[["customer_id", "city", "age", "gender", "tenure"]], on="customer_id", how="left")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    df_cohort["lapsed"] = (~df_cohort["customer_id"].isin(active_sept_cids)).astype(int)

    n_cohort = len(df_cohort)
    n_lapsed = int(df_cohort["lapsed"].sum())
    pct_lapsed = (n_lapsed / n_cohort) * 100
    print(f"Cohort size: {n_cohort} | Lapsed: {n_lapsed} ({pct_lapsed:.2f}%)")
    assert n_cohort == 1156 and n_lapsed == 443

    feature_cols = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure",
        "city",
        "gender",
        "fav_category",
    ]
    assert df_cohort[feature_cols].isna().sum().sum() == 0, "Missing values found in feature columns!"

    X = df_cohort[feature_cols].copy()
    y = df_cohort["lapsed"].values
    cids = df_cohort["customer_id"].values

    X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(
        X, y, cids, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    assert len(X_train) == 924 and len(X_test) == 232

    return df_cohort, X_train, X_test, y_train, y_test, cid_train, cid_test


def train_and_evaluate(df_cohort, X_train, X_test, y_train, y_test, cid_test, rfm_seg):
    print("\n=== Task 5: Preprocess Without Leakage ===")
    num_cols = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount_pct",
        "cancellation_rate",
        "avg_delivery_minutes",
        "age",
        "tenure",
    ]
    cat_cols = ["city", "gender", "fav_category"]

    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
        ]
    )
    preprocessor.fit(X_train)
    freq_idx = num_cols.index("frequency")
    train_freq_mean = preprocessor.named_transformers_["num"].mean_[freq_idx]
    cohort_freq_mean = df_cohort["frequency"].mean()
    print(
        f"Learned train scaler mean of frequency: {train_freq_mean:.4f} | "
        f"Full-cohort mean of frequency: {cohort_freq_mean:.4f}"
    )

    print("\n=== Task 6: Train, Tune and Evaluate ===")
    # (a) Baseline: always predicting "not lapsed" (0)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_base = dummy.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base, zero_division=0)
    print(f"Baseline (predict not lapsed): Test Accuracy={base_acc*100:.1f}%, Test F1={base_f1:.4f}")

    # (b) Tune LogisticRegression
    pipe_lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    grid_lr = GridSearchCV(
        pipe_lr,
        param_grid={"clf__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
    )
    grid_lr.fit(X_train, y_train)
    print(f"LogisticRegression Best Params: {grid_lr.best_params_}, Best CV F1: {grid_lr.best_score_:.4f}")

    y_pred_lr = grid_lr.predict(X_test)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    p_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    r_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
    print(f"LogisticRegression Test: Precision={p_lr:.4f}, Recall={r_lr:.4f}, F1={f1_lr:.4f}")
    print("LogisticRegression Confusion Matrix:\n", cm_lr)

    # (c) Tune RandomForestClassifier
    pipe_rf = Pipeline(
        [("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]
    )
    grid_rf = GridSearchCV(
        pipe_rf,
        param_grid={"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
        cv=5,
        scoring="f1",
    )
    grid_rf.fit(X_train, y_train)
    print(f"RandomForest Best Params: {grid_rf.best_params_}, Best CV F1: {grid_rf.best_score_:.4f}")

    y_pred_rf = grid_rf.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    p_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    r_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
    print(f"RandomForest Test: Precision={p_rf:.4f}, Recall={r_rf:.4f}, F1={f1_rf:.4f}")
    print("RandomForest Confusion Matrix:\n", cm_rf)
    assert f1_rf >= 0.55 and f1_lr >= 0.55

    print("\n=== Task 7: Class-Balance Experiment ===")
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    n_to_add = n_neg - n_pos
    pos_indices = np.where(y_train == 1)[0]
    rng = np.random.RandomState(42)
    extra_pos_idx = rng.choice(pos_indices, size=n_to_add, replace=True)
    bal_idx = np.concatenate([np.arange(len(X_train)), extra_pos_idx])

    X_train_bal = X_train.iloc[bal_idx].reset_index(drop=True)
    y_train_bal = y_train[bal_idx]
    print(
        f"Balanced training counts: class 0 = {(y_train_bal == 0).sum()}, class 1 = {(y_train_bal == 1).sum()}"
    )
    assert (y_train_bal == 0).sum() == 570 and (y_train_bal == 1).sum() == 570

    best_c = grid_lr.best_params_["clf__C"]
    pipe_lr_bal = Pipeline(
        [("prep", preprocessor), ("clf", LogisticRegression(C=best_c, max_iter=1000, random_state=42))]
    )
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_bal = pipe_lr_bal.predict(X_test)
    p_bal = precision_score(y_test, y_pred_bal, zero_division=0)
    r_bal = recall_score(y_test, y_pred_bal, zero_division=0)
    f1_bal = f1_score(y_test, y_pred_bal, zero_division=0)
    print(f"Balanced LogisticRegression Test: Precision={p_bal:.4f}, Recall={r_bal:.4f}, F1={f1_bal:.4f}")

    print("\n=== Task 8: Interpret, Persist and Use ===")
    best_rf_pipe = grid_rf.best_estimator_
    best_lr_pipe = grid_lr.best_estimator_

    feat_names = best_rf_pipe.named_steps["prep"].get_feature_names_out()
    rf_imps = best_rf_pipe.named_steps["clf"].feature_importances_
    top5_rf = sorted(zip(feat_names, rf_imps), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 Random Forest Feature Importances:")
    for fname, imp in top5_rf:
        print(f"  {fname}: {imp:.4f}")

    lr_coefs = best_lr_pipe.named_steps["clf"].coef_[0]
    coef_map = dict(zip(feat_names, lr_coefs))
    print("Selected Logistic Regression Coefficients:")
    for key in ["num__frequency", "num__recency", "num__monetary", "num__tenure"]:
        if key in coef_map:
            print(f"  {key}: {coef_map[key]:+.4f}")

    # Persist selected model (best Random Forest) and verify reload
    model_path = "outputs/lapse_model.joblib"
    joblib.dump(best_rf_pipe, model_path)
    reloaded_model = joblib.load(model_path)
    orig_preds = best_rf_pipe.predict(X_test)
    reload_preds = reloaded_model.predict(X_test)
    assert np.array_equal(orig_preds, reload_preds), "Reloaded model predictions do not match!"
    print("Saved and verified outputs/lapse_model.joblib (exact prediction match).")

    # Retention list: test-set customers in Premium segment, ranked by predicted lapse probability
    probs_test = reloaded_model.predict_proba(X_test)[:, 1]
    test_df = pd.DataFrame(
        {
            "customer_id": cid_test,
            "lapse_probability": probs_test.round(4),
            "actually_lapsed": y_test,
        }
    )
    prem_cids = set(rfm_seg.loc[rfm_seg["is_premium"], "customer_id"])
    ret_df = (
        test_df[test_df["customer_id"].isin(prem_cids)]
        .sort_values(["lapse_probability", "customer_id"], ascending=[False, True])
        .head(20)
        .reset_index(drop=True)
    )
    ret_df.to_csv("outputs/retention_list.csv", index=False)
    print(
        f"Wrote outputs/retention_list.csv (rows={len(ret_df)}, truly lapsed={int(ret_df['actually_lapsed'].sum())})"
    )
    print(ret_df)


def main():
    orders_df = pd.read_csv("data/clean/orders.csv")
    customers_df = pd.read_csv("data/clean/customers.csv")
    products_df = pd.read_csv("data/clean/products.csv")

    rfm = build_rfm(orders_df)
    rfm_seg, profile, evr = cluster_and_profile(rfm)
    df_cohort, X_train, X_test, y_train, y_test, cid_train, cid_test = build_lapse_cohort(
        orders_df, customers_df, products_df
    )
    train_and_evaluate(df_cohort, X_train, X_test, y_train, y_test, cid_test, rfm_seg)


if __name__ == "__main__":
    main()
