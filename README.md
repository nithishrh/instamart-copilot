# Instamart Customer Intelligence & Support Copilot

## Overview

This repository contains the complete **Customer Intelligence & Support Copilot** built for Swiggy Instamart across six Indian metro cities (Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, and Pune). All monetary values are expressed in Indian Rupees (INR), and all language-model interactions execute deterministically offline through `mock_llm.py`.

The repository is structured into four connected parts:
- **Part A (`part_a.py`, `sql/queries.sql`)**: Audits and cleans the raw synthetic exports (`customers.csv`, `products.csv`, `orders.csv`), writes validated tables to `data/clean/` and `data/instamart.db`, executes seven analytical SQL queries, and cross-checks results in pandas.
- **Part B (`part_b.py`)**: Builds the snapshot RFM table, clusters customers with K-Means (`K=2..8`) and PCA (`outputs/customer_segments.csv`), defines a leak-free pre-cutoff lapse prediction cohort, tunes `LogisticRegression` and `RandomForestClassifier` pipelines (`outputs/lapse_model.joblib`), tests class-balance oversampling, and outputs `outputs/retention_list.csv`.
- **Part C (`part_c.py`)**: Implements support ticket triage with input guardrails, regex PII redaction, exponential-backoff retries, JSON and keyword output validation, token-budget enforcement (`outputs/triage_budget_demo.csv`), and segment-aware routing (`outputs/triage_results.csv`).
- **Part D (`part_d.py`)**: Chunks the five policy documents in `data/policies/`, builds a TF-IDF retrieval-augmented generation (RAG) policy assistant, evaluates overlap=10 versus overlap=0 grounding, and runs a resumable ReAct refund agent with five tools, three safety stop guards, checkpoint files in `state/`, run logs in `outputs/agent_runs.jsonl`, and financial entries in `outputs/refund_ledger.jsonl`.

## Reproduce

Run these numbered commands in order from a clean clone of the `main` branch:

```bash
# 1. Create and activate an isolated Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install required open-source packages
pip install -r requirements.txt

# 3. Generate deterministic raw synthetic data and policy documents
python3 generate_data.py

# 4. Run Part A: Audit, clean, store to SQLite, validate checkpoints, and run SQL & pandas analyses
python3 part_a.py

# 5. Run Part B: Build RFM segments, fit PCA, train/tune lapse models, and export retention list
python3 part_b.py

# 6. Run Part C: Redact PII, classify support tickets with retries/validation, and route by segment
python3 part_c.py

# 7. Run Part D: Evaluate Policy RAG assistant and execute ReAct refund agent with safety & resume demos
python3 part_d.py
```

## Part A

### Task 2: Raw Data Audit

#### Table Shapes, Data Types, and Missing Values (Standard pandas Load)

| Table | Shape | Column | Data Type | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- | ---: | ---: |
| `customers.csv` | `(1200, 5)` | `customer_id` | `object` | 0 | 0.00% |
| `customers.csv` | `(1200, 5)` | `city` | `object` | 0 | 0.00% |
| `customers.csv` | `(1200, 5)` | `signup_date` | `object` | 0 | 0.00% |
| `customers.csv` | `(1200, 5)` | `age` | `float64` | 61 | 5.08% |
| `customers.csv` | `(1200, 5)` | `gender` | `object` | 30 | 2.50% |
| `products.csv` | `(40, 4)` | `product_id` | `object` | 0 | 0.00% |
| `products.csv` | `(40, 4)` | `product_name` | `object` | 0 | 0.00% |
| `products.csv` | `(40, 4)` | `category` | `object` | 0 | 0.00% |
| `products.csv` | `(40, 4)` | `price_inr` | `int64` | 0 | 0.00% |
| `orders.csv` | `(13084, 9)` | `order_id` | `object` | 0 | 0.00% |
| `orders.csv` | `(13084, 9)` | `customer_id` | `object` | 0 | 0.00% |
| `orders.csv` | `(13084, 9)` | `product_id` | `object` | 0 | 0.00% |
| `orders.csv` | `(13084, 9)` | `quantity` | `int64` | 0 | 0.00% |
| `orders.csv` | `(13084, 9)` | `order_ts` | `object` | 0 | 0.00% |
| `orders.csv` | `(13084, 9)` | `status` | `object` | 0 | 0.00% |
| `orders.csv` | `(13084, 9)` | `discount_pct` | `int64` | 0 | 0.00% |
| `orders.csv` | `(13084, 9)` | `order_amount` | `object` | 53 (`"N/A"` parsed as NaN; 67 `"error"` strings remain) | 0.41% |
| `orders.csv` | `(13084, 9)` | `delivery_minutes` | `float64` | 2,541 | 19.42% |

#### Specific Raw Audit Figures (`keep_default_na=False`)

| Audit Metric | Table | Reported Value | Reference Checkpoint |
| :--- | :--- | ---: | ---: |
| Total raw order rows | `orders.csv` | 13,084 | 13,084 |
| Exact duplicate rows | `orders.csv` | 119 | 119 |
| Rows with `status == 'pending'` | `orders.csv` | 656 | 656 |
| Distinct `order_id` values | `orders.csv` | 12,309 | 12,309 |
| Non-numeric `order_amount` entries (`"N/A"` + `"error"`) | `orders.csv` | 120 | 120 |
| Negative `delivery_minutes` entries (`-10`) | `orders.csv` | 106 | 106 |
| Blank `delivery_minutes` entries | `orders.csv` | 2,541 | 2,541 |
| Blank `age` entries | `customers.csv` | 61 | 61 |
| `age` values outside 10–100 (`-4, 0, 150, 200, 280, 999`) | `customers.csv` | 6 | 6 |
| Distinct raw `city` spellings | `customers.csv` | 24 | 24 |
| Distinct non-blank raw `gender` spellings | `customers.csv` | 10 | 10 |
| Blank `gender` entries | `customers.csv` | 30 | 30 |

### Task 3: Clean `customers`

Documented cleaning rules and values changed:
1. **Standardise `city` (`str.strip().str.title()`)**: Modified **476** cell values across 1,200 rows, collapsing 24 raw spellings into exactly 6 canonical cities: **Pune (227)**, **Delhi (210)**, **Bengaluru (197)**, **Chennai (191)**, **Hyderabad (191)**, and **Mumbai (184)**.
2. **Standardise `gender`**: Mapped all 10 non-blank spelling variants and 30 blanks to `male`, `female`, or `unknown`, modifying **950** cell values and resulting in **female: 610**, **male: 560**, and **unknown: 30**.
3. **Clean and impute `age`**: Replaced **6** impossible ages outside `10–100` with `NaN`, then imputed all **67** missing ages (61 originally blank + 6 out-of-range) using the median age of the customer's city via `groupby("city")["age"].transform("median")`. Result: **0 missing ages**, minimum age **18.0**, maximum age **60.0**.
4. **Convert `signup_date` to datetime**: Parsed all **1,200** `signup_date` strings to `datetime64[ns]`.

### Task 4: Clean `orders`

Documented cleaning rules executed in exact sequence:
1. **Rule i — Convert `order_ts` to datetime**: Parsed all **13,084** `order_ts` strings to `datetime64[ns]`.
2. **Rule ii — Remove exact duplicate rows**: Dropped **119** exact duplicate rows, reducing the table from 13,084 to 12,965 rows.
3. **Rule iii — Deduplicate by `order_id` keeping latest `order_ts`**: Sorted by `order_ts` and kept the latest record per `order_id`, removing **656** earlier `pending` rows and leaving **12,309** unique orders (**11,110 `completed`**, **1,199 `cancelled`**, **0 `pending`**).
4. **Rule iv — Convert `order_amount` to numeric and recompute missing**: Coerced **112** unparseable strings (`"N/A"` and `"error"`) among the 12,309 deduplicated orders to `NaN` and imputed all **112** values using `round(price_inr * quantity * (1 - discount_pct / 100), 2)`. Result: **0 missing `order_amount`**, and total `order_amount` over completed orders is **INR 2,998,668.80**.
5. **Rule v — Clean and impute `delivery_minutes` by product category for completed orders only**: Replaced **105** negative `delivery_minutes` values (`-10`) with `NaN`, imputed all **774** missing `delivery_minutes` among `completed` orders (669 blank + 105 negative) with the product category median among completed orders (`groupby` + `transform`), and left all **1,199** `cancelled` orders as missing (`NaN`). Result: **0 missing `delivery_minutes` among completed orders** and **1,199 missing among cancelled orders**.

### Task 5: Store and Validate

The three cleaned tables are saved to `data/clean/customers.csv`, `data/clean/products.csv`, and `data/clean/orders.csv`, and loaded into `data/instamart.db` (`customers`, `products`, `orders`). Running `part_a.py` verifies all reference assertions and prints `ALL CHECKPOINTS PASSED`.

### Task 6: SQL Analysis (`sql/queries.sql`)

| Query | Description | Reported Result |
| :--- | :--- | :--- |
| **Q1** | `WHERE` (`status = 'completed' AND order_amount > 500`) then `HAVING SUM(order_amount) > 8000` | **30 customers** (top: `C0693` at INR 13,776.55 across 6 high-value orders) |
| **Q2** | Three-table join: completed revenue by `city` and `category` | **30 rows**; total revenue **INR 2,998,668.80**; largest cell is **Bengaluru / Electronics** at **INR 233,909.45** (191 orders) |
| **Q3** | Scalar subquery: products priced above catalogue average (INR 434.25) | **11 products** (all 8 Electronics items plus Ghee 500ml, Body Lotion 400ml, Sunscreen 50g) |
| **Q4** | `LEFT JOIN`: customers with no completed order | **25 customers** (1,200 total minus 1,175 with completed orders) |
| **Q5** | CTE: customers with completed spend above the per-customer average | **550 customers**; average spend **INR 2,552.06**; top customer **`C0693` at INR 14,766.55** |
| **Q6** | Window functions (`RANK()`, `DENSE_RANK()`, `ROW_NUMBER()`) partitioned by `city` | **1,175 rows**; top per city listed below; exactly **8 rows** have `RANK()` different from `ROW_NUMBER()` |
| **Q7** | Daily completed revenue and running cumulative total (`SUM(revenue) OVER (ORDER BY day)`) | **183 daily rows** (`2025-04-01` to `2025-09-30`); final cumulative revenue **INR 2,998,668.80** |

#### Q6 Top Customer per City

| City | Top Customer ID | Completed Spend (INR) |
| :--- | :--- | ---: |
| Bengaluru | `C0082` | INR 12,856.20 |
| Chennai | `C0930` | INR 10,151.40 |
| Delhi | `C0291` | INR 13,742.60 |
| Hyderabad | `C0743` | INR 12,730.05 |
| Mumbai | `C0693` | INR 14,766.55 |
| Pune | `C0426` | INR 10,377.10 |

**Why ties cause `RANK()` to differ from `ROW_NUMBER()`**: When two or more customers in the same city have the exact same `total_spend`, `RANK()` assigns the same rank to all tied rows, whereas `ROW_NUMBER()` assigns a distinct sequential integer (`1, 2, 3, ...`) to every row regardless of ties, causing the second (and any subsequent) tied customer in each of the 8 tied pairs to have a larger `ROW_NUMBER()` than `RANK()`.

### Task 7: pandas Cross-Check and Exploratory Analysis

#### (a) Q2 Pivot Table Cross-Check (Revenue in INR) and Per-City Aggregation

| City | Dairy (INR) | Electronics (INR) | Grocery (INR) | Personal Care (INR) | Snacks (INR) | City Total Revenue (INR) | Mean Order Amount (INR) | Completed Orders |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bengaluru | 87,075.05 | 233,909.45 | 87,053.65 | 44,055.85 | 71,549.40 | **523,643.40** | 276.47 | 1,894 |
| Chennai | 78,301.80 | 195,508.80 | 82,782.10 | 38,710.75 | 68,325.70 | **463,629.15** | 272.72 | 1,700 |
| Delhi | 117,271.70 | 158,549.10 | 95,436.35 | 40,693.20 | 90,879.45 | **502,829.80** | 243.62 | 2,064 |
| Hyderabad | 83,731.00 | 228,913.10 | 90,702.05 | 43,573.40 | 81,670.35 | **528,589.90** | 284.95 | 1,855 |
| Mumbai | 75,209.85 | 214,738.40 | 76,770.85 | 40,217.30 | 72,122.10 | **479,058.50** | 297.00 | 1,613 |
| Pune | 90,349.55 | 187,629.25 | 95,049.75 | 47,682.00 | 80,207.50 | **500,918.05** | 252.48 | 1,984 |
| **All** | **531,938.95** | **1,219,248.10** | **527,794.75** | **254,932.50** | **464,754.50** | **2,998,668.80** | **269.91** | **11,110** |

#### (b) Completed `order_amount` Distribution Statistics & Correlation Matrix

- **Summary Statistics (`N = 11,110` completed orders)**: Mean = **INR 269.91**, Median = **INR 135.00**, Skewness = **4.61**, Q1 = **INR 64.80**, Q3 = **INR 297.00**, IQR = **INR 232.20**, Lower 1.5×IQR Fence = **INR −283.50** (**0 outliers below**), Upper 1.5×IQR Fence = **INR 645.30** (**897 outliers above**).
- **Correlation Matrix (Completed Orders)**:
  - `corr(order_amount, discount_pct) = -0.0951` (largest in absolute value)
  - `corr(order_amount, quantity) = +0.0873`
  - `corr(quantity, delivery_minutes) = -0.0126`
  - `corr(discount_pct, delivery_minutes) = -0.0098`
  - `corr(order_amount, delivery_minutes) = -0.0031`
  - `corr(discount_pct, quantity) = -0.0004`

**Exploratory Analysis (158 words)**:
This dataset captures six months (`2025-04-01` to `2025-09-30`) of cleaned Swiggy Instamart transactions across 1,200 customers, 40 catalogue products in five categories, and 12,309 orders (11,110 completed) in six Indian metros. Completed `order_amount` is strongly right-skewed (skewness 4.61), with a mean of INR 269.91 that is nearly double the median of INR 135.00, reflecting frequent everyday grocery, dairy, and snack purchases alongside occasional expensive electronics orders. Using the 1.5×IQR rule (Q1 INR 64.80, Q3 INR 297.00, IQR INR 232.20), there are 0 low outliers below the lower fence (INR −283.50) and 897 high-value outliers above the upper fence (INR 645.30), all driven by genuine Electronics baskets rather than data corruption. Across `order_amount`, `discount_pct`, `quantity`, and `delivery_minutes`, linear relationships are weak: the pair with the largest correlation in absolute value is `order_amount` with `discount_pct` at `−0.0951` (slightly larger in magnitude than `order_amount` with `quantity` at `+0.0873`), and correlation of this size does not establish causation.

## Part B

### Task 1: RFM Table Summary (Snapshot `2025-10-01 00:00:00`, `N = 1,175` Customers)

| Metric | Mean | Median | Minimum | Maximum |
| :--- | ---: | ---: | ---: | ---: |
| `recency` (days) | 35.12 | 19.0 | 0 | 180 |
| `frequency` (completed orders) | 9.46 | 6.0 | 1 | 33 |
| `monetary` (INR) | 2,552.06 | 1,781.90 | 14.40 | 14,766.55 |

### Task 2: K-Means Clustering (`K = 2..8`)

| K | Inertia | Silhouette Score |
| ---: | ---: | ---: |
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| **4** | **743.1** | **0.526** |
| 5 | 617.3 | 0.479 |
| 6 | 503.2 | 0.483 |
| 7 | 408.7 | 0.420 |
| 8 | 366.7 | 0.410 |

**Selected K = 4**. Inertia decreases sharply from `K=2` (1,820.8) to `K=4` (743.1) before flattening into a gradual slope, and `K=4` achieves the global maximum silhouette score (`0.526`) while yielding four distinct, actionable customer archetypes. The inertia of `K=8` is mathematically guaranteed to be lower than that of `K=2` because adding more centroids reduces the average squared distance from each point to its nearest centroid (reaching zero if `K` equals the number of distinct points). Consequently, raw inertia always favours larger `K` values and cannot pick `K` on its own without elbow diminishing-returns analysis, silhouette separation, and business interpretability.

### Task 3: Segment Profiles (`K = 4`) and PCA Projection

*Note: Cluster IDs (`segment_id` 0–3) are arbitrary numerical labels produced by the K-Means initialisation with no intrinsic ordering.*

| `segment_id` | Segment Name | Size | Mean Recency (Days) | Mean Frequency | Mean Monetary (INR) | `is_premium` | Recommended Business Action |
| ---: | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `0` | **Premium Big-Ticket Buyers** | 138 | 24.12 | 6.77 | **7,129.70** | `True` | Offer priority VIP support, early access to electronics launches, and bundled protection plans. |
| `3` | **Loyal Frequent Shoppers** | 419 | 8.77 | 19.40 | 3,410.02 | `False` | Enrol in a monthly subscription pass with free-delivery rewards to lock in habitual grocery spend. |
| `1` | **Active Casual Buyers** | 436 | 28.86 | 3.77 | 1,057.62 | `False` | Send personalised basket-threshold nudges (free delivery at INR 199+) to grow order frequency. |
| `2` | **At-Risk Dormant Buyers** | 182 | 109.04 | 2.20 | 685.99 | `False` | Trigger automated win-back discount vouchers on staple groceries before permanent churn. |

- **PCA on Scaled RFM (`PCA(n_components=2)`)**: `explained_variance_ratio_` is **PC1 = 0.634** and **PC2 = 0.218**, capturing **0.852 (85.2%)** of total variance across the three RFM dimensions.
- Saved to `outputs/customer_segments.csv` (`1,175` rows).

### Task 4 & Task 5: Leak-Free Lapse Cohort and Preprocessing (`Cutoff 2025-09-01`)

- **Cohort & Target**: **1,156 customers** placed at least one completed order prior to `2025-09-01`. Of these, **443 customers (38.32%)** placed no completed order from `2025-09-01` through `2025-09-30` (`lapsed = 1`).
- **Strict Anti-Leakage Discipline**: No data dated on or after `2025-09-01` entered any feature calculation, and the whole-window segment labels from Task 3 were excluded from the feature matrix. All encoders (`OneHotEncoder(drop="first")`) and scalers (`StandardScaler()`) were fitted strictly on the **924 training rows** (never on the full cohort or the **232 test rows**), and no resampling was applied to the test set.
- **Frequency Mean Check**: The fitted training-set `StandardScaler` learned a `frequency` mean of **8.1147**, whereas the full-cohort `frequency` mean is **8.0285**; they differ because the scaler is fitted exclusively on the 80% stratified training sample (924 customers) rather than all 1,156 customers, avoiding test-set distribution leakage. All 11 feature columns have **0 missing values**.

### Task 6: Model Training, GridSearchCV Tuning, and Test Evaluation

| Model | Best Hyperparameters (`cv=5`, `scoring="f1"`) | Best CV F1 | Test Precision (Lapsed) | Test Recall (Lapsed) | Test F1 (Lapsed) | Test Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **Baseline (`DummyClassifier`)** | Always predict `0` (not lapsed) | 0.0000 | 0.0000 | 0.0000 | **0.0000** (Acc: 61.6%) | `[[143, 0], [89, 0]]` |
| **`LogisticRegression`** | `C = 1` | 0.6565 | 0.6444 | 0.6517 | **0.6480** | `[[111, 32], [31, 58]]` |
| **`RandomForestClassifier`** | `max_depth = 3, min_samples_leaf = 20` | **0.6827** | 0.6444 | 0.6517 | **0.6480** | `[[111, 32], [31, 58]]` |

**Model Selection Justification**: We select the tuned **`RandomForestClassifier`** (`max_depth=3, min_samples_leaf=20`) as our production model because it achieved the highest 5-fold cross-validated F1 score (`0.6827` vs `0.6565` for Logistic Regression) and tied on the untouched test set with an F1 of `0.6480` (`precision = 0.6444`, `recall = 0.6517`, well above the `0.55` threshold). On the 232-customer test set, the model correctly identifies 58 of 89 lapsing customers while generating 32 false positives and 31 false negatives. In quick-commerce retention, the cost of missing a lapsing customer (a false negative, losing thousands of INR in lifetime gross merchandise value) is substantially higher than the cost of wrongly targeting a loyal customer (a false positive, spending a small promotional coupon or push notification). Because the shallow random forest captures non-linear interactions between recency, frequency, and monetary spend with higher cross-validation stability, it provides a reliable probability ranking for retention outreach.

### Task 7: Class-Balance Oversampling Experiment

| Logistic Regression Variant (`C = 1`) | Training Class Counts (`0 / 1`) | Test Precision | Test Recall | Test F1 |
| :--- | :--- | ---: | ---: | ---: |
| **Unbalanced Training Set** | `570 / 354` | 0.6444 | 0.6517 | 0.6480 |
| **Balanced Training Set (Oversampled)** | `570 / 570` | 0.5923 | **0.8652** | **0.7032** |

**Impact for the Retention Team**: **Recall** moved the most, jumping by **+21.35 percentage points** (from `0.6517` to `0.8652`), while precision stayed near `0.60` (dipping slightly from `0.6444` to `0.5923`), lifting overall test F1 to `0.7032`. For the retention team, oversampling shifts the decision boundary so the campaign catches nearly 87% of genuinely lapsing customers at the modest cost of sending retention nudges to a slightly wider group of active shoppers.

### Task 8: Feature Interpretation, Model Persistence, and Retention List

#### Top 5 Random Forest Feature Importances (`feature_importances_`)

1. **`num__frequency` (`0.3095`)**: Customers who have placed fewer completed orders prior to the cutoff are much more likely to lapse in September.
2. **`num__monetary` (`0.3003`)**: Cumulative historical spend strongly separates occasional high-ticket electronics buyers from habitual weekly grocery shoppers.
3. **`num__recency` (`0.1487`)**: A longer gap in days since the customer's last pre-cutoff purchase indicates declining engagement and higher lapse risk.
4. **`num__cancellation_rate` (`0.0707`)**: Higher historical order cancellation rates signal friction or dissatisfaction associated with subsequent churn.
5. **`num__avg_discount_pct` (`0.0406`)**: Promotion sensitivity differentiates deal-driven occasional buyers from steady everyday buyers.

#### Selected Logistic Regression Coefficients (`C = 1`)

- **`num__frequency` (`-1.6274`, negative)**: Each standard-deviation increase in pre-cutoff order frequency sharply reduces the log-odds of lapsing, confirming habitual ordering as the strongest retention anchor.
- **`num__recency` (`+0.0466`, positive)**: More days elapsed since the customer's last completed order increases the log-odds of lapsing.
- **`num__tenure` (`-0.0489`, negative)**: Longer-tenured customers on the platform have slightly lower log-odds of lapsing than newly acquired customers.

#### Model Persistence and Premium Retention Target List (`outputs/retention_list.csv`)

- Saved `outputs/lapse_model.joblib` with `joblib.dump`, reloaded with `joblib.load`, and verified 100% identical predictions across all 232 test customers.
- Ranked test-set customers flagged as `is_premium == True` by predicted lapse probability and exported the top **20 customers** to `outputs/retention_list.csv`; among these top 20 Premium test customers, **3 truly lapsed** (`C0188`, `C0271`, `C0525`).
- **Leakage Caveat**: Because the `Premium` segment flag comes from Task 3's whole-window RFM clustering (`2025-04-01` through `2025-09-30`), which includes the September window that defines the lapse label, it is a post-hoc whole-window label, making `outputs/retention_list.csv` an illustrative targeting list rather than a leakage-free evaluation.

## Part C

### Tasks 1 & 2: Input Guardrails and PII Redaction Across 164 Tickets

- **Input Guardrails**: Across the 164 raw tickets in `data/raw/tickets.csv`, **4 empty tickets** (`T0157`–`T0160`) received status `skipped_empty` and **4 overly long tickets** (`T0161`–`T0164`, each estimating to **734 tokens** `> MAX_INPUT_TOKENS = 400`) received status `flagged_long`; none of these 8 tickets triggered a model call.
- **PII Redaction Counts**: Applying the three regular expressions in exact order across all 164 tickets replaced **55 emails** (`[EMAIL]`), **91 phone numbers** (`[PHONE]`), and **132 order IDs** (`[ORDER_ID]`), for **278 total redactions**, with **0 matches remaining** when re-applying the patterns to the redacted texts.

### Tasks 4 & 5: Resilient Retries and Output Validation

- **Retries**: Across a full run over the **156 eligible tickets**, `mock_llm.chat` raised exactly **19 `TransientLLMError`s** (every 9th call), and all **19 failures were recovered** on retry `n = 0` with jittered exponential backoff waits between `0.045s` and `0.055s` (`0.045s <= wait <= 0.055s`).
- **Validation Status Breakdown (156 Eligible Tickets)**:
  - **Rule-validated and routed automatically (147 tickets)**: **135 `classified_json`** (valid JSON with an allowed category) and **12 `classified_keyword`** (non-JSON chatty replies recovered via `re.findall(r"[a-z]+", text.lower())` matching a single allowed category word).
  - **Routed to human review (`human_review`, 17 tickets across full 164-ticket file)**: **9 `flagged_invalid_output`** (valid JSON whose category `"shipping"` is outside the allowed list), plus the **4 `skipped_empty`** and **4 `flagged_long`** tickets.

### Task 6: Zero-Shot vs. Few-Shot Comparison (No Token Budget)

| Prompt Strategy | Eligible Tickets | Classified Tickets (`json` + `keyword`) | Correct vs `true_category` | Classification Accuracy | Total Tokens Used | Tokens per Classified Ticket |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Zero-Shot** | 156 | 147 (`135 + 12`) | 126 / 147 | **85.7%** | 10,960 | **74.56** |
| **Few-Shot** | 156 | 147 (`135 + 12`) | 147 / 147 | **100.0%** | 18,916 | **128.68** |

#### Zero-Shot Confusion Table (147 Classified Tickets)

| `true_category` \ Predicted | `billing` | `delivery` | `fraud` | `technical` | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`billing`** | 38 | 0 | 0 | 0 | 38 |
| **`delivery`** | 0 | 36 | 0 | 0 | 36 |
| **`fraud`** | **11** | 0 | 23 | 0 | 34 |
| **`technical`** | **10** | 0 | 0 | 29 | 39 |
| **Total** | 59 | 36 | 23 | 29 | 147 |

**When Extra Few-Shot Tokens Are Worth Paying For**: Adding labeled examples in the few-shot prompt increases token consumption from 10,960 to 18,916 total tokens across the batch, costing an extra **54.12 tokens per classified ticket** (`128.68` vs `74.56`). However, that modest token premium eliminates all 21 zero-shot misclassifications, raising accuracy from **85.7% (126/147)** to **100.0% (147/147)**. Crucially, 11 of the zero-shot errors were `fraud` tickets (`"I was charged for order..."`) misrouted as `billing` because `billing` preceded `fraud` before the few-shot examples reordered the classifier's priority, alongside 10 `technical` payment-page errors misclassified as `billing`. Because missing a fraud ticket breaches the 2-hour security SLA and exposes the company to financial loss, paying 54.12 extra tokens per ticket for few-shot disambiguation is overwhelmingly worthwhile in production.

### Task 7: Token Budget Enforcement and Segment-Aware Routing

- **Budget Demo (`outputs/triage_budget_demo.csv`)**: Running the zero-shot pipeline with `TOKEN_BUDGET = 6000` processed 86 eligible tickets (accumulating **6,059 tokens**) before exhausting the budget, marking the remaining **70 eligible tickets** as `budget_exhausted` (routed to `human_review`, `NORMAL` priority).
- **Final Few-Shot Triage Output (`outputs/triage_results.csv`, `TOKEN_BUDGET = 100000`, 164 rows, 18,916 total tokens)**:
  - `technical_queue` (`NORMAL`): **35 tickets**
  - `security_team` (`HIGH`): **34 tickets** (all 34 classified `fraud` tickets)
  - `billing_queue` (`NORMAL`): **34 tickets**
  - `delivery_queue` (`NORMAL`): **33 tickets**
  - `human_review` (`NORMAL`): **17 tickets** (`4 skipped_empty` + `4 flagged_long` + `9 flagged_invalid_output`)
  - `premium_desk` (`HIGH`): **11 tickets** (non-fraud classified tickets from `is_premium == True` customers)

## Part D

### Tasks 1 & 2: Policy Document Chunking and TF-IDF Retrieval

- **Chunking (`40 words`, `overlap = 10 words`)**: Produced **15 chunks in total** across the five policy files:
  - `refund_policy.txt`: **4 chunks**
  - `delivery_policy.txt`: **3 chunks**
  - `escalation_policy.txt`: **3 chunks**
  - `payment_policy.txt`: **3 chunks**
  - `privacy_policy.txt`: **2 chunks**
- **Query Check (`"refund policy late delivery"`)**: All top 3 retrieved chunks come from **`refund_policy.txt`** (cosine similarities `0.2719`, `0.2514`, and `0.2454`).
- **TF-IDF Explanation and Limitation**: Term Frequency (TF) measures how often a word appears inside a specific policy chunk, while Inverse Document Frequency (IDF) down-weights common words that appear across many chunks and boosts rare, discriminative terms. A core limitation of TF-IDF for policy retrieval is that it relies purely on exact lexical overlap and cannot match synonyms or paraphrases (for example, a customer asking about *"delayed rider compensation"* will have zero token overlap with *"late-delivery refund"*).

### Tasks 3 & 4: Retrieval and Grounding Evaluation (`overlap = 10` vs. `overlap = 0`)

| Chunking Configuration | Total Chunks | Answerable Retrieval (`expected_doc` in Top 3) | Answerable Grounding (`expected_fact` in Answer) | Unanswerable Fallback (`"I could not find..."`) |
| :--- | ---: | ---: | ---: | ---: |
| **40 words, overlap = 10** | 15 | **12 / 12** | **12 / 12** | **2 / 2** |
| **40 words, overlap = 0** | 13 | **12 / 12** | **10 / 12** | **2 / 2** |

**Why Zero Overlap (`overlap = 0`) Breaks Grounding**:
With `overlap = 0`, two answerable questions fail fact grounding because the fixed 40-word boundary slices a policy sentence right between its condition and its fact:
1. **`"What delivery time makes a completed order eligible for a full refund?"` (expects `"more than 45 minutes"`)**: In `refund_policy.txt`, chunk 1 (words 0–39) ends right at *"If a completed order is delivered in more than 45 minutes, the customer"*, while chunk 2 (words 40–79) begins with *"receives a full refund of the order amount."* When the answerer selects the clause matching *"eligible for a full refund"*, it returns only `"receives a full refund of the order amount."`, omitting `"more than 45 minutes"`.
2. **`"How soon is money returned when a payment fails but is deducted?"` (expects `"48 hours"`)**: In `payment_policy.txt`, chunk 1 (words 0–39) ends at *"If a payment fails but money is deducted, the amount is returned automatically"*, while chunk 2 starts with *"within 48 hours."*, severing the time window from the condition.

### Tasks 5 & 6: ReAct Refund Agent Results (`data/raw/agent_scenarios.csv`)

| Scenario ID | Customer ID | Final Status | Refund Amount (INR) | Expected Outcome | Expected Refund (INR) | Model Calls (`llm_calls`) | Tokens Used (`tokens_used`) | Tools Executed |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | :--- |
| **`S1`** | `C0025` | `REFUND` | **128.25** | `REFUND` | 128.25 | 5 | 992 | `get_order`, `search_policy`, `calculate_refund`, `issue_refund` |
| **`S2`** | `C0567` | `REFUND` | **135.00** | `REFUND` | 135.00 | 5 | 992 | `get_order`, `search_policy`, `calculate_refund`, `issue_refund` |
| **`S3`** | `C0070` | `ESCALATE` | **1,999.00** | `ESCALATE` | 1,999.00 | 5 | 979 | `get_order`, `search_policy`, `calculate_refund`, `escalate_to_human` |
| **`S4`** | `C0462` | `DENIED` | **0.00** | `DENIED` | 0.00 | 4 | 681 | `get_order`, `search_policy`, `calculate_refund` |
| **`S5`** | `C0758` | `DENIED` | **0.00** | `DENIED` | 0.00 | 3 | 394 | `get_order`, `search_policy` |
| **`S6`** | `C0001` | `NOT_FOUND` | **0.00** | `NOT_FOUND` | 0.00 | 2 | 110 | `get_order` |
| **`S7`** | `C0025` | `FALLBACK_MAX_ITERATIONS` | **0.00** | `FALLBACK` | 0.00 | 6 | 3,114 | `search_policy` (×6) |

### Task 7: Stop-Safely Demonstrations

| Demonstration Run | Configured Limit | Resulting Status | Model Calls (`llm_calls`) | Tokens Used (`tokens_used`) | Tool Steps Completed |
| :--- | :--- | :--- | ---: | ---: | ---: |
| **Max-Iterations Guard (`S7`)** | `max_iterations = 6` (`token_limit = 6000`) | **`FALLBACK_MAX_ITERATIONS`** | 6 | 3,114 | 6 |
| **Token-Limit Guard (`S1` demo)** | `token_limit = 200` | **`FALLBACK_TOKEN_LIMIT`** | 3 | 411 | 2 |
| **Wall-Clock Timeout Guard (`S1` demo)** | `time_limit_seconds = 0` | **`FALLBACK_TIMEOUT`** | 0 | 0 | 0 |

### Task 8: Crash-and-Resume Demonstration on `S1`

- **Before Simulated Crash (`run_agent` with crash after tool #2)**: Executed `get_order` (step 1) and `search_policy` (step 2), saved checkpoint `state/s1_crash_resume.json`, and raised `RuntimeError`.
- **After `resume_agent("s1_crash_resume")`**: Loaded checkpoint state with 2 prior tool observations and executed `calculate_refund` (step 3) and `issue_refund` (step 4), finishing with status **`REFUND`** and amount **INR 128.25** in **5 total model calls**.
- **Ordered List of Tools Executed Across Crash and Resume**:
  1. `get_order`
  2. `search_policy`
  3. `calculate_refund`
  4. `issue_refund`
  *(Exactly 4 tool executions, none repeated.)*

### Task 9: ReAct vs. Plan-and-Execute Comparison (194 words)

In our ReAct refund agent, the model reasons step-by-step after each observation, requiring **5 model calls (992 tokens)** for a full refund (`S1`/`S2`) or escalation (`S3`), **4 calls (681 tokens)** for an on-time denial (`S4`), **3 calls (394 tokens)** for a cancelled-order denial (`S5`), and **2 calls (110 tokens)** when an order does not exist (`S6`). By contrast, a Plan-and-Execute architecture generates the entire multi-step tool sequence in 1 upfront planning call and uses 1 final synthesis call (2 model calls total on the happy path), significantly reducing token overhead when every step succeeds as assumed. However, Plan-and-Execute handles unexpected observations poorly: when `get_order` returns `NOT_FOUND` (`S6`) or `status = "cancelled"` (`S5`), a static plan blindly attempts downstream steps (`calculate_refund`, `issue_refund`) on invalid data or triggers an expensive replanning call, whereas ReAct immediately short-circuits after the observation (stopping in just 2 calls / 110 tokens on `S6` and 3 calls / 394 tokens on `S5`). For this financial refund workflow, **we choose ReAct** because each tool decision strictly depends on the previous database or policy observation, and preventing invalid payouts outweighs saving 2–3 lightweight model calls.

## Limitations and Next Steps

1. **Illustrative Retention List (Post-Hoc Segment Label)**: Because the `Premium` segment flag in Part B comes from the full six-month RFM table (`2025-04-01` through `2025-09-30`), which includes the September window that defines the lapse label, it is a post-hoc whole-window label, meaning `outputs/retention_list.csv` is an illustrative targeting list and not a leakage-free evaluation. In production, customer segments used to filter retention campaigns must be computed strictly as of the pre-prediction cutoff date (`2025-09-01`).
2. **Lexical Matching in Policy RAG**: TF-IDF retrieval relies on exact word stems and misses semantic paraphrases; upgrading to dense sentence embeddings combined with sentence-aware chunking would improve retrieval robustness.
3. **Rule-Enforced Refund Date Window**: While all simulated refund scenarios occur within the 7-day eligibility window, a production refund agent should compare the ticket timestamp against `order_ts` in `calculate_refund` to enforce the 7-day policy automatically.
