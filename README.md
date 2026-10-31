# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements an end-to-end **Customer Intelligence & Support Copilot** for Swiggy Instamart across six Indian metro cities (Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, and Pune). All monetary values are expressed in Indian Rupees (INR), and all language-model interactions execute deterministically offline through `mock_llm.py`.

The project is organized into four connected modules:

* **Part A (`part_a.py`, `sql/queries.sql`)**: Audits and cleans raw synthetic exports (`customers.csv`, `products.csv`, `orders.csv`), writes validated tables to `data/clean/` and `data/instamart.db`, executes seven analytical SQL queries, and cross-checks results in pandas.

* **Part B (`part_b.py`)**: Builds the snapshot RFM table, clusters customers with K-Means and PCA (`outputs/customer_segments.csv`), defines a leak-free pre-cutoff lapse prediction cohort, tunes `LogisticRegression` and `RandomForestClassifier` pipelines (`outputs/lapse_model.joblib`), evaluates class-balance oversampling, and exports a priority retention list (`outputs/retention_list.csv`).

* **Part C (`part_c.py`)**: Implements support ticket triage with input guardrails, regex PII redaction, jittered exponential-backoff retries, JSON and keyword output validation, token-budget enforcement (`outputs/triage_budget_demo.csv`), and segment-aware routing (`outputs/triage_results.csv`).

* **Part D (`part_d.py`)**: Chunks the five policy documents in `data/policies/`, builds a TF-IDF retrieval-augmented generation (RAG) policy assistant, compares chunk overlap strategies, and runs a crash-resumable ReAct refund agent with five tools, three safety stop guards, state checkpoints (`state/`), run logs (`outputs/agent_runs.jsonl`), and financial ledger entries (`outputs/refund_ledger.jsonl`).

## Project Structure

* `generate_data.py`: Generates deterministic synthetic raw CSV datasets (`data/raw/`) and policy documents (`data/policies/`).

* `mock_llm.py`: Offline deterministic stand-in for a hosted LLM chat API supporting classification, policy QA, and ReAct tool-calling.

* `part_a.py` & `sql/queries.sql`: Data auditing, cleaning, SQLite loading, SQL analytics, and exploratory data analysis.

* `part_b.py`: RFM segmentation, K-Means/PCA clustering, leak-free churn modeling, and retention targeting.

* `part_c.py`: Support ticket guardrails, PII redaction, retry resilience, prompt evaluation, and queue routing.

* `part_d.py`: TF-IDF policy RAG pipeline, ReAct refund agent, safety stop guards, and checkpoint recovery.

* `requirements.txt`: Pinned open-source Python dependencies (`pandas`, `numpy`, `scikit-learn`, `scipy`, `joblib`).

## Reproduce

Run these numbered commands in order from a clean clone of the `main` branch:

```
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

## Part A: Data Foundation & Exploratory Analysis

### Cleaning `customers.csv` (`1,200` rows)

1. **Standardise `city`**: Strips whitespace and applies title-casing (`str.strip().str.title()`), collapsing 24 raw spelling variants into the 6 canonical metro cities.

2. **Standardise `gender`**: Maps all 10 non-blank spelling variants and blank entries into `male`, `female`, or `unknown`.

3. **Clean and impute `age`**: Replaces impossible ages outside `10–100` with `NaN` and imputes all missing values using the median age of the customer's city (`groupby("city")["age"].transform("median")`).

4. **Parse `signup_date`**: Converts `signup_date` strings to `datetime64[ns]`.

### Cleaning `orders.csv` (`13,084` raw rows → `12,309` unique orders)

1. **Parse `order_ts`**: Converts order timestamps to `datetime64[ns]`.

2. **Drop exact duplicates**: Removes exact duplicate rows prior to order-level deduplication.

3. **Deduplicate by `order_id`**: Sorts by `order_ts` and retains the latest record per `order_id`, eliminating earlier `pending` rows and leaving only final `completed` (`11,110`) and `cancelled` (`1,199`) orders.

4. **Recompute missing `order_amount`**: Coerces unparseable strings (`"N/A"`, `"error"`) to `NaN` and recomputes them from catalogue prices via `round(price_inr * quantity * (1 - discount_pct / 100), 2)`.

5. **Clean and impute `delivery_minutes`**: Replaces negative durations (`-10`) with `NaN`, imputes missing durations for `completed` orders using the product category median among completed orders, and leaves `cancelled` orders as `NaN`.

### SQL & Exploratory Findings

* **Validation**: Cleaned tables are saved to `data/clean/*.csv` and `data/instamart.db`, passing all integrity assertions in `part_a.py`.

* **Window Function Ties (`Q6`)**: When ranking customer spend per city, `RANK()` assigns identical ranks to tied customers, whereas `ROW_NUMBER()` assigns distinct sequential integers regardless of ties.

* **Distribution & Correlations**: Completed `order_amount` (six-month total **INR 2,998,668.80**) is strongly right-skewed (skewness `4.61`, mean **INR 269.91** vs. median **INR 135.00**), driven by high-value Electronics baskets above the upper 1.5×IQR fence (**INR 645.30**) rather than data corruption. Linear correlations across `order_amount`, `discount_pct`, `quantity`, and `delivery_minutes` are weak (`|r| < 0.10`).

## Part B: Customer Segmentation & Lapse Prediction

### RFM Clustering & PCA (`outputs/customer_segments.csv`)

* **Snapshot RFM (`2025-10-01`)**: Aggregates recency (days since last completed order), frequency (completed order count), and monetary spend (INR) across all `1,175` customers with completed orders.

* **Choosing `K = 4`**: Inertia drops sharply from `K=2` to `K=4` before flattening, and `K=4` achieves the global maximum silhouette score (`0.526`). Because raw inertia mathematically decreases with every added centroid, combining elbow diminishing returns, silhouette separation, and business interpretability is required to select `K`.

* **PCA Projection**: Two principal components capture **85.2%** of total variance (`PC1 = 63.4%`, `PC2 = 21.8%`).

* **Four Customer Archetypes**:

  * **Premium Big-Ticket Buyers (`is_premium = True`)**: High monetary spend (**INR 7,129.70** mean); target with VIP support, early electronics access, and bundled protection plans.

  * **Loyal Frequent Shoppers**: High order frequency (`19.40` mean orders, `8.77` days recency); lock in habitual spend with monthly free-delivery subscription passes.

  * **Active Casual Buyers**: Moderate recency and small baskets; send basket-threshold nudges (free delivery at INR 199+) to grow order frequency.

  * **At-Risk Dormant Buyers**: Long inactivity (`109.04` days mean recency); trigger automated win-back discount vouchers on staple groceries.

### Leak-Free Lapse Modeling (`outputs/lapse_model.joblib`)

* **Cohort & Anti-Leakage Discipline**: Evaluates the `1,156` customers with at least one completed order before `2025-09-01`, predicting whether they placed zero completed orders in September (`38.32%` lapsed). All features are computed strictly from pre-cutoff transactions, whole-window segment labels are excluded, and `StandardScaler` / `OneHotEncoder` transformers are fitted exclusively on the `924` training rows (never on the `232` test rows).

* **Model Selection**: Tuned `RandomForestClassifier` (`max_depth=3, min_samples_leaf=20`) is selected for production because it achieves the highest 5-fold cross-validated F1 (`0.6827` vs. `0.6565` for `LogisticRegression`) and achieves a test F1 of `0.6480` (`precision = 0.6444`, `recall = 0.6517`). In quick-commerce retention, the lifetime GMV lost from a false negative far outweighs the small coupon cost of a false positive.

* **Class-Balance Oversampling**: Oversampling the minority lapsed class in training boosts `LogisticRegression` test recall by **+21.35 percentage points** (`0.6517` → `0.8652`) and lifts test F1 to `0.7032`, catching nearly 87% of lapsing shoppers.

* **Top Feature Drivers**: Pre-cutoff `frequency`, `monetary` spend, `recency`, `cancellation_rate`, and `avg_discount_pct` are the strongest predictors of churn.

## Part C: Support Ticket Triage with Guardrails

* **Input Guardrails & PII Redaction**: Before any model call, empty tickets are marked `skipped_empty` and tickets exceeding `MAX_INPUT_TOKENS = 400` are marked `flagged_long`. Eligible tickets are scrubbed via ordered regular expressions replacing emails (`[EMAIL]`), 10-digit Indian mobile numbers (`[PHONE]`), and order IDs (`[ORDER_ID]`) with zero residual PII leaks.

* **Resilient Retries & Validation**: Transient API failures (`TransientLLMError`) are recovered automatically using exponential backoff with uniform jitter (`BASE_DELAY = 0.05s`). Model replies are validated via strict JSON parsing (`classified_json`) with a single-keyword fallback (`classified_keyword`) for conversational responses; out-of-schema categories (such as `"shipping"`) are caught as `flagged_invalid_output`.

* **Zero-Shot vs. Few-Shot Trade-Off**: Few-shot prompting adds roughly `54` tokens per ticket (`128.68` vs. `74.56` tokens) but raises classification accuracy on valid tickets from **85.7%** to **100.0%**, eliminating critical zero-shot errors where `fraud` and `technical` tickets were misclassified as `billing`.

* **Budgeting & Routing (`outputs/triage_results.csv`)**: Enforces hard token budgets (`budget_exhausted` fallback) and routes classified tickets by risk and customer tier: `fraud` tickets go to `security_team` (`HIGH` priority), non-fraud tickets from `is_premium` customers go to `premium_desk` (`HIGH` priority), standard tickets go to their category queue (`NORMAL` priority), and unclassifiable tickets go to `human_review` (`NORMAL` priority).

## Part D: Policy RAG Assistant & ReAct Refund Agent

### Policy RAG Assistant

* **Chunking & Retrieval**: Splits the five policy documents in `data/policies/` into 40-word chunks with a 10-word overlap (`15` chunks total) and retrieves the top-3 chunks via TF-IDF cosine similarity.

* **Why Overlap Matters**: Using `overlap = 10` achieves **100% (12/12)** retrieval and fact grounding on answerable policy questions (plus **2/2** safe fallbacks on unanswerable questions). Setting `overlap = 0` drops fact grounding to **10/12** because fixed 40-word boundaries slice sentences right between conditions and timeframes (such as separating *"more than 45 minutes"* from *"full refund"*, and *"payment fails"* from *"within 48 hours"*).

### ReAct Refund Agent & Safety Architecture

* **Five Agent Tools**:

  1. `get_order`: Queries order status, amount, and delivery minutes from `data/instamart.db`.

  2. `search_policy`: Retrieves top-3 policy chunks via TF-IDF similarity.

  3. `calculate_refund`: Computes eligible late-delivery refund amounts (`delivery_minutes > 45`).

  4. `issue_refund`: Approves automatic refunds up to INR 1,000 and appends to `outputs/refund_ledger.jsonl`.

  5. `escalate_to_human`: Escalates eligible refunds above INR 1,000 to a human support lead and logs the entry.

* **Three Safety Stop Guards**: Prevents runaway loops and cost overruns by enforcing `max_iterations = 6` (`FALLBACK_MAX_ITERATIONS`), `token_limit` (`FALLBACK_TOKEN_LIMIT`), and `time_limit_seconds` (`FALLBACK_TIMEOUT`).

* **Crash-and-Resume Checkpointing**: Persists agent state to `state/<run_id>.json` after every tool execution so interrupted runs (`resume_agent`) pick up from the exact next step without re-running completed database or ledger tools.

* **Why ReAct over Plan-and-Execute**: Because each refund decision depends strictly on the previous database or policy observation (for example, immediately halting when an order is `NOT_FOUND` or `cancelled`), ReAct short-circuits invalid requests in 2–3 lightweight calls rather than blindly executing or replanning a static multi-step sequence.

## Limitations and Next Steps

1. **Pre-Cutoff Segmentation for Retention Campaigns**: In Part B, the `is_premium` flag used to filter `outputs/retention_list.csv` comes from the full six-month RFM table (`2025-04-01` to `2025-09-30`), making that target list illustrative; production retention pipelines should compute segment labels strictly prior to the prediction cutoff date (`2025-09-01`).

2. **Semantic Embeddings for Policy RAG**: TF-IDF relies on exact lexical overlap and misses synonyms or paraphrases; replacing TF-IDF with dense sentence embeddings and sentence-boundary chunking will improve retrieval robustness.

3. **Automated 7-Day Refund Window Check**: While all simulated refund requests occur within the 7-day policy window, production deployment should compare the support ticket timestamp against `order_ts` inside ``calculate_refund` to enforce the 7-day policy automatically.
