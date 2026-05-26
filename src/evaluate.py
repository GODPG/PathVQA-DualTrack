import re
import json
import math
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd


ARTICLES = {"a", "an", "the"}
PUNCT_RE = re.compile(r"[^a-z0-9\s]")

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def is_closed(answer_type):
    return str(answer_type).strip().lower() in {"yes/no", "closed", "binary"}


def normalize_answer(text, yes_no=False):
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()

    if yes_no:
        if re.search(r"\byes\b", text):
            return "yes"
        if re.search(r"\bno\b", text):
            return "no"

    text = text.replace("/", " ")
    text = PUNCT_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in ARTICLES]
    tokens = [NUMBER_WORDS.get(t, t) for t in tokens]
    return " ".join(tokens).strip()


def token_list(text):
    norm = normalize_answer(text)
    return norm.split()


def token_f1(pred, gold):
    pred_tokens = token_list(pred)
    gold_tokens = token_list(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_precision(pred, gold):
    pred_tokens = token_list(pred)
    gold_tokens = token_list(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    return overlap / len(pred_tokens)


def token_recall(pred, gold):
    pred_tokens = token_list(pred)
    gold_tokens = token_list(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    return overlap / len(gold_tokens)


def relaxed_correct(pred_norm, gold_norm, question=""):
    if not pred_norm or not gold_norm:
        return False

    if pred_norm == gold_norm:
        return True

    if pred_norm in gold_norm or gold_norm in pred_norm:
        return True

    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())

    if pred_tokens and gold_tokens:
        overlap = len(pred_tokens & gold_tokens)
        shorter_ratio = overlap / max(1, min(len(pred_tokens), len(gold_tokens)))
        if shorter_ratio >= 0.6:
            return True

    if token_f1(pred_norm, gold_norm) >= 0.5:
        return True

    return False


def bleu_score_sentence(pred, gold, max_n=4):
    pred_tokens = token_list(pred)
    gold_tokens = token_list(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    weights = [1.0 / max_n] * max_n
    precisions = []

    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1))
        gold_ngrams = Counter(tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens) - n + 1))

        if not pred_ngrams:
            precisions.append(1e-9)
            continue

        overlap = sum((pred_ngrams & gold_ngrams).values())
        total = sum(pred_ngrams.values())

        # Add-one smoothing to avoid all-zero BLEU for short answers.
        precisions.append((overlap + 1) / (total + 1))

    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))

    if len(pred_tokens) > len(gold_tokens):
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - len(gold_tokens) / max(1, len(pred_tokens)))

    return brevity_penalty * math.exp(log_precision)


def lcs_len(a, b):
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)

    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr

    return prev[-1]


def rouge_l_sentence(pred, gold):
    pred_tokens = token_list(pred)
    gold_tokens = token_list(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = lcs_len(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    beta = 1.2
    return ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)


def word_overlap_semantic_score(pred, gold):
    """
    WBSS-like score.
    In the original literature, WBSS may use word embeddings.
    Here we use normalized token F1 as a reproducible fallback without external models.
    """
    return token_f1(pred, gold)


def clinical_based_similarity_score(pred, gold):
    """
    CBSS-like score.
    This lightweight version rewards exact/containment and token overlap.
    It is deterministic and does not require external clinical ontology packages.
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if not pred_norm or not gold_norm:
        return 0.0

    if pred_norm == gold_norm:
        return 1.0

    if pred_norm in gold_norm or gold_norm in pred_norm:
        return 0.8

    return token_f1(pred_norm, gold_norm)


def try_bertscore(preds, refs):
    try:
        from bert_score import score
    except Exception as e:
        return None, f"bert_score is not installed: {e}"

    try:
        _, _, f1 = score(
            preds,
            refs,
            lang="en",
            verbose=False,
            rescale_with_baseline=False,
        )
        return [float(x) for x in f1.tolist()], None
    except Exception as e:
        return None, str(e)


def pct(x):
    if x is None:
        return None
    if pd.isna(x):
        return None
    return round(100 * float(x), 2)


def mean_or_none(series):
    if len(series) == 0:
        return None
    return float(series.mean())


def summarize(df, group_cols):
    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        total = len(g)
        closed = g[g["is_closed"]]
        open_df = g[~g["is_closed"]]

        row = dict(zip(group_cols, keys))
        row["total_samples"] = total
        row["closed_samples"] = len(closed)
        row["open_samples"] = len(open_df)

        row["closed_accuracy"] = pct(mean_or_none(closed["exact_correct"])) if len(closed) else None
        row["open_exact_accuracy"] = pct(mean_or_none(open_df["exact_correct"])) if len(open_df) else None
        row["open_relaxed_accuracy"] = pct(mean_or_none(open_df["relaxed_correct"])) if len(open_df) else None
        row["overall_exact_accuracy"] = pct(mean_or_none(g["exact_correct"])) if total else None
        row["overall_relaxed_accuracy"] = pct(mean_or_none(g["relaxed_correct"])) if total else None

        row["token_precision"] = pct(mean_or_none(g["token_precision"])) if total else None
        row["token_recall"] = pct(mean_or_none(g["token_recall"])) if total else None
        row["token_f1"] = pct(mean_or_none(g["token_f1"])) if total else None

        row["open_token_precision"] = pct(mean_or_none(open_df["token_precision"])) if len(open_df) else None
        row["open_token_recall"] = pct(mean_or_none(open_df["token_recall"])) if len(open_df) else None
        row["open_token_f1"] = pct(mean_or_none(open_df["token_f1"])) if len(open_df) else None

        row["open_bleu"] = pct(mean_or_none(open_df["bleu"])) if len(open_df) else None
        row["open_rouge_l"] = pct(mean_or_none(open_df["rouge_l"])) if len(open_df) else None
        row["open_wbss"] = pct(mean_or_none(open_df["wbss"])) if len(open_df) else None
        row["open_cbss"] = pct(mean_or_none(open_df["cbss"])) if len(open_df) else None

        if "bertscore_f1" in open_df.columns and len(open_df):
            row["open_bertscore_f1"] = pct(mean_or_none(open_df["bertscore_f1"]))
        else:
            row["open_bertscore_f1"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Format: ModelName=/path/result.csv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore if bert-score is installed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for spec in args.inputs:
        if "=" not in spec:
            raise ValueError(f"Invalid input spec: {spec}. Use ModelName=/path/file.csv")

        model_name, path = spec.split("=", 1)
        df = pd.read_csv(path)

        if "model" not in df.columns:
            df["model"] = model_name
        else:
            df["model"] = df["model"].fillna(model_name)

        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)

    if "prediction" not in df.columns:
        if "prediction_norm" in df.columns:
            df["prediction"] = df["prediction_norm"]
        elif "raw_prediction" in df.columns:
            df["prediction"] = df["raw_prediction"]
        elif "prediction_raw" in df.columns:
            df["prediction"] = df["prediction_raw"]
        else:
            raise ValueError("No prediction column found.")

    df["is_closed"] = df["answer_type"].apply(is_closed)

    pred_norms = []
    answer_norms = []
    exacts = []
    relaxed = []
    precisions = []
    recalls = []
    f1s = []
    bleus = []
    rouge_ls = []
    wbss_scores = []
    cbss_scores = []

    for _, row in df.iterrows():
        yes_no = bool(row["is_closed"])
        pred = row.get("prediction", "")
        answer = row.get("answer", "")

        pred_norm = normalize_answer(pred, yes_no=yes_no)
        answer_norm = normalize_answer(answer, yes_no=yes_no)

        exact = pred_norm == answer_norm
        rel = exact if yes_no else relaxed_correct(
            pred_norm,
            answer_norm,
            question=row.get("question", ""),
        )

        pred_norms.append(pred_norm)
        answer_norms.append(answer_norm)
        exacts.append(exact)
        relaxed.append(rel)

        precisions.append(token_precision(pred_norm, answer_norm))
        recalls.append(token_recall(pred_norm, answer_norm))
        f1s.append(token_f1(pred_norm, answer_norm))

        if yes_no:
            bleus.append(None)
            rouge_ls.append(None)
            wbss_scores.append(None)
            cbss_scores.append(None)
        else:
            bleus.append(bleu_score_sentence(pred_norm, answer_norm))
            rouge_ls.append(rouge_l_sentence(pred_norm, answer_norm))
            wbss_scores.append(word_overlap_semantic_score(pred_norm, answer_norm))
            cbss_scores.append(clinical_based_similarity_score(pred_norm, answer_norm))

    df["prediction_norm"] = pred_norms
    df["answer_norm"] = answer_norms
    df["exact_correct"] = exacts
    df["relaxed_correct"] = relaxed
    df["token_precision"] = precisions
    df["token_recall"] = recalls
    df["token_f1"] = f1s
    df["bleu"] = bleus
    df["rouge_l"] = rouge_ls
    df["wbss"] = wbss_scores
    df["cbss"] = cbss_scores

    bertscore_error = None
    if args.bertscore:
        open_mask = ~df["is_closed"]
        preds = df.loc[open_mask, "prediction_norm"].fillna("").astype(str).tolist()
        refs = df.loc[open_mask, "answer_norm"].fillna("").astype(str).tolist()

        scores, err = try_bertscore(preds, refs)
        if scores is None:
            bertscore_error = err
            df["bertscore_f1"] = None
        else:
            df["bertscore_f1"] = None
            df.loc[open_mask, "bertscore_f1"] = scores
    else:
        df["bertscore_f1"] = None
        bertscore_error = "BERTScore was not requested. Use --bertscore to enable it."

    scored_path = out_dir / "scored_predictions.csv"
    df.to_csv(scored_path, index=False, encoding="utf-8-sig")

    by_model = summarize(df, ["model"]).sort_values("overall_relaxed_accuracy", ascending=False)
    by_answer_type = summarize(df, ["model", "answer_type"])
    by_question_type = summarize(df, ["model", "question_type"])

    by_model.to_csv(out_dir / "metrics_by_model.csv", index=False, encoding="utf-8-sig")
    by_answer_type.to_csv(out_dir / "metrics_by_answer_type.csv", index=False, encoding="utf-8-sig")
    by_question_type.to_csv(out_dir / "metrics_by_question_type.csv", index=False, encoding="utf-8-sig")

    metrics_json = {
        "metrics_by_model": by_model.to_dict("records"),
        "total_rows": int(len(df)),
        "open_bertscore_error": bertscore_error,
        "notes": {
            "unit": "percentage for accuracy and similarity metrics in CSV/JSON outputs",
            "wbss": "implemented as normalized token-F1 fallback without external word embeddings",
            "cbss": "implemented as exact/containment/token-overlap clinical similarity fallback without external ontology",
            "bertscore": "optional; requires bert-score and downloadable/local encoder model",
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)

    print("\nMetrics by model:")
    print(by_model.to_string(index=False))
    print("\nSaved to:", out_dir)
    if bertscore_error:
        print("BERTScore:", bertscore_error)


if __name__ == "__main__":
    main()