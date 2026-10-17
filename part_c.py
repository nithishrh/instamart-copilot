"""Part C — Support Ticket Triage with Guardrails (22 marks)
Implements input guardrails (empty and long ticket checks), regex PII redaction (email, phone,
order ID), resilient mock_llm calls with exponential backoff and jitter, output validation
(JSON + keyword fallback), zero-shot vs. few-shot evaluation, token-budget enforcement, and
customer-segment-aware routing.
"""
import json
import os
import random
import re
import time
import pandas as pd
import mock_llm

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED_CATS = {"billing", "delivery", "technical", "fraud"}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

ZERO_SHOT_PROMPT = (
    "You are a support ticket classifier. CLASSIFY each support ticket into one of the four "
    "allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only of the form {"category": "...", "reason": "..."}.'
)

FEW_SHOT_PROMPT = (
    "You are a support ticket classifier. CLASSIFY each support ticket into one of the four "
    "allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only of the form {"category": "...", "reason": "..."}.\n'
    "EXAMPLES:\n"
    'Ticket: "There is an unauthorized transaction on my card linked to [ORDER_ID]."\n'
    'Output: {"category": "fraud", "reason": "unauthorized transaction reported"}\n'
    'Ticket: "The app crashes every time I open the cart for [ORDER_ID]."\n'
    'Output: {"category": "technical", "reason": "app crash error"}'
)


def redact_text(text):
    text, n_email = EMAIL_RE.subn("[EMAIL]", text)
    text, n_phone = PHONE_RE.subn("[PHONE]", text)
    text, n_order = ORDER_RE.subn("[ORDER_ID]", text)
    return text, n_email, n_phone, n_order


def verify_redaction_on_all(tickets_df):
    print("=== Task 2: Redact Personal Information ===")
    tot_email = tot_phone = tot_order = 0
    redacted_texts = []
    for txt in tickets_df["text"]:
        r_txt, ne, np_, no = redact_text(txt)
        tot_email += ne
        tot_phone += np_
        tot_order += no
        redacted_texts.append(r_txt)

    tot_all = tot_email + tot_phone + tot_order
    print(
        f"Redactions: email={tot_email}, phone={tot_phone}, order_id={tot_order}, total={tot_all}"
    )
    assert tot_email == 55 and tot_phone == 91 and tot_order == 132 and tot_all == 278

    # Show by code that re-applying the three patterns to the redacted texts finds nothing
    leftover = 0
    for r_txt in redacted_texts:
        _, ne, np_, no = redact_text(r_txt)
        leftover += ne + np_ + no
    print(f"Leftover matches on re-application: {leftover}")
    assert leftover == 0
    return redacted_texts


def call_with_retry(ticket_id, messages, rng):
    for n in range(3):
        try:
            return mock_llm.chat(messages), n
        except mock_llm.TransientLLMError:
            wait = BASE_DELAY * (2 ** n) * rng.uniform(0.9, 1.1)
            print(f"  [Retry] ticket_id={ticket_id} retry={n} wait={wait:.4f}s")
            time.sleep(wait)
    return mock_llm.chat(messages), 3


def validate_output(reply_text):
    try:
        data = json.loads(reply_text)
        if isinstance(data, dict) and data.get("category") in ALLOWED_CATS:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (reply_text or "").lower()))
        matched = ALLOWED_CATS & words
        if len(matched) == 1:
            return "classified_keyword", next(iter(matched))
        return "flagged_invalid_output", ""


def route_ticket(status, category, is_premium):
    if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return f"{category}_queue", "NORMAL"


def run_triage_pipeline(tickets_df, seg_map, system_prompt, token_budget=None, verbose_retries=False):
    mock_llm.reset()
    rng = random.Random(42)

    running_tokens = 0
    total_retries = 0
    records = []

    for _, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_txt = row["text"]
        true_cat = row["true_category"]

        seg_info = seg_map.get(cid, {"segment_name": "", "is_premium": False})
        seg_name = seg_info["segment_name"]
        is_prem = seg_info["is_premium"]

        # Task 1: Input guardrails
        if raw_txt.strip() == "":
            status, cat, tok = "skipped_empty", "", 0
        elif mock_llm.est_tokens(raw_txt) > MAX_INPUT_TOKENS:
            status, cat, tok = "flagged_long", "", 0
        elif token_budget is not None and running_tokens >= token_budget:
            status, cat, tok = "budget_exhausted", "", 0
        else:
            red_txt, _, _, _ = redact_text(raw_txt)
            # Assert no PII or raw text issues before calling model
            assert EMAIL_RE.search(red_txt) is None
            assert PHONE_RE.search(red_txt) is None
            assert ORDER_RE.search(red_txt) is None

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": red_txt},
            ]
            # Resilient call with up to 3 retries
            for n in range(3):
                try:
                    resp = mock_llm.chat(messages)
                    break
                except mock_llm.TransientLLMError:
                    wait = BASE_DELAY * (2 ** n) * rng.uniform(0.9, 1.1)
                    assert 0.045 <= wait <= 0.220
                    total_retries += 1
                    if verbose_retries:
                        print(f"  [Retry] ticket_id={tid} retry={n} wait={wait:.5f}s")
                    time.sleep(wait)
            else:
                resp = mock_llm.chat(messages)

            tok = resp.usage.total_tokens
            running_tokens += tok
            reply_content = resp.choices[0].message.content
            status, cat = validate_output(reply_content)

        route, priority = route_ticket(status, cat, is_prem)
        records.append(
            {
                "ticket_id": tid,
                "customer_id": cid,
                "status": status,
                "category": cat,
                "segment_name": seg_name,
                "priority": priority,
                "route": route,
                "tokens_used": tok,
                "true_category": true_cat,
            }
        )

    res_df = pd.DataFrame(records)
    return res_df, total_retries, running_tokens


def main():
    tickets_df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    seg_df = pd.read_csv("outputs/customer_segments.csv")
    seg_map = {
        r["customer_id"]: {"segment_name": r["segment_name"], "is_premium": bool(r["is_premium"])}
        for _, r in seg_df.iterrows()
    }

    print("=== Task 1: Input Guardrails Check ===")
    empty_cnt = sum(t.strip() == "" for t in tickets_df["text"])
    long_toks = [mock_llm.est_tokens(t) for t in tickets_df["text"] if mock_llm.est_tokens(t) > MAX_INPUT_TOKENS]
    print(f"Empty tickets: {empty_cnt} | Long tickets: {len(long_toks)} (tokens={long_toks})")
    assert empty_cnt == 4 and len(long_toks) == 4 and all(t == 734 for t in long_toks)

    verify_redaction_on_all(tickets_df)

    print("\n=== Tasks 3, 4, 5, 6: Zero-Shot Full Run (No Budget) ===")
    zs_df, zs_retries, zs_tokens = run_triage_pipeline(
        tickets_df, seg_map, ZERO_SHOT_PROMPT, token_budget=None, verbose_retries=True
    )
    print(f"Zero-Shot Retries recovered: {zs_retries}")
    print("Zero-Shot Status Counts:\n", zs_df["status"].value_counts().to_dict())
    assert zs_retries == 19
    zs_status = zs_df["status"].value_counts().to_dict()
    assert zs_status["classified_json"] == 135
    assert zs_status["classified_keyword"] == 12
    assert zs_status["flagged_invalid_output"] == 9

    zs_classified = zs_df[zs_df["status"].isin(["classified_json", "classified_keyword"])]
    zs_correct = (zs_classified["category"] == zs_classified["true_category"]).sum()
    zs_acc = (zs_correct / len(zs_classified)) * 100
    print(
        f"Zero-Shot Classified: {len(zs_classified)} | Correct: {zs_correct} ({zs_acc:.1f}%) | "
        f"Total Tokens: {zs_tokens} | Tokens/Classified: {zs_tokens / len(zs_classified):.2f}"
    )
    assert len(zs_classified) == 147 and zs_correct == 126
    print(
        "\nZero-Shot Confusion Table (true_category vs predicted category):\n",
        pd.crosstab(zs_classified["true_category"], zs_classified["category"], margins=True),
    )

    print("\n=== Task 6: Few-Shot Full Run (No Budget) ===")
    fs_df, fs_retries, fs_tokens = run_triage_pipeline(
        tickets_df, seg_map, FEW_SHOT_PROMPT, token_budget=None, verbose_retries=False
    )
    fs_classified = fs_df[fs_df["status"].isin(["classified_json", "classified_keyword"])]
    fs_correct = (fs_classified["category"] == fs_classified["true_category"]).sum()
    fs_acc = (fs_correct / len(fs_classified)) * 100
    print(
        f"Few-Shot Classified: {len(fs_classified)} | Correct: {fs_correct} ({fs_acc:.1f}%) | "
        f"Total Tokens: {fs_tokens} | Tokens/Classified: {fs_tokens / len(fs_classified):.2f}"
    )
    assert len(fs_classified) == 147 and fs_correct == 147

    print("\n=== Task 7: Budget, Routing and Output ===")
    # (a) Zero-shot with TOKEN_BUDGET = 6000
    demo_budget = 6000
    demo_df, _, demo_tokens = run_triage_pipeline(
        tickets_df, seg_map, ZERO_SHOT_PROMPT, token_budget=demo_budget, verbose_retries=False
    )
    exhausted_cnt = int((demo_df["status"] == "budget_exhausted").sum())
    print(f"Budget Demo (TOKEN_BUDGET={demo_budget}): total_tokens={demo_tokens}, budget_exhausted={exhausted_cnt}")
    assert exhausted_cnt >= 1

    out_cols = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]
    os.makedirs("outputs", exist_ok=True)
    demo_df[out_cols].to_csv("outputs/triage_budget_demo.csv", index=False)
    print("Wrote outputs/triage_budget_demo.csv")

    # (b) Final outputs/triage_results.csv from few-shot run with TOKEN_BUDGET = 100000
    final_df, _, final_tokens = run_triage_pipeline(
        tickets_df, seg_map, FEW_SHOT_PROMPT, token_budget=100000, verbose_retries=False
    )
    final_df[out_cols].to_csv("outputs/triage_results.csv", index=False)
    print(f"Wrote outputs/triage_results.csv (rows={len(final_df)}, total_tokens={final_tokens})")
    print("Final Route Counts:\n", final_df["route"].value_counts().to_dict())
    print("Final Priority Counts:\n", final_df["priority"].value_counts().to_dict())


if __name__ == "__main__":
    main()
