import json
import csv
from difflib import SequenceMatcher
import math
import re
import string
from collections import Counter
from pathlib import Path


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_PUNCT_EXCEPT_HYPHEN_SLASH = "".join(ch for ch in string.punctuation if ch not in {"-", "/"})
_PUNCT_EXCEPT_HYPHEN_SLASH_TABLE = str.maketrans("", "", _PUNCT_EXCEPT_HYPHEN_SLASH)
_YES_NO_RE = re.compile(r"\b(yes|no|true|false|y|n|yeah|yep|nope)\b", flags=re.IGNORECASE)
_OPEN_PREFIXES = (
    "the image shows",
    "this image shows",
    "image shows",
    "based on the image",
    "based on image",
    "it appears to be",
    "it appears",
    "likely representing",
    "likely",
    "final answer:",
    "answer:",
    "the answer is",
)
_OPEN_LEADING_PHRASES = (
    "it is",
    "this is",
    "there is",
    "there are",
)
_DESCRIPTION_WORDS = {"image", "show", "shows", "shown", "based", "appears", "appearing", "likely"}
_CANDIDATE_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[\).]|[A-Za-z][\).])\s*(.+?)\s*$")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
_SYNONYM_PATH = Path("configs/pathvqa_answer_synonyms.json")
_ANSWER_SYNONYMS = None
DEFAULT_GENERATIVE_CONFIG = {
    "hybrid_token_f1_threshold": 0.5,
    "hybrid_char_similarity_threshold": 0.85,
    "hybrid_bertscore_threshold": 0.85,
    "partial_token_f1_threshold": 0.3,
    "partial_char_similarity_threshold": 0.75,
    "partial_word_similarity_threshold": 0.5,
    "soft_floor": 0.8,
    "overlong_prediction_tokens": 20,
}


def generative_config(**overrides):
    config = dict(DEFAULT_GENERATIVE_CONFIG)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def load_answer_synonyms(path=_SYNONYM_PATH):
    if not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        normalize_basic_answer(key): normalize_basic_answer(value)
        for key, value in data.items()
        if str(key).strip() and str(value).strip()
    }


def answer_synonyms():
    global _ANSWER_SYNONYMS
    if _ANSWER_SYNONYMS is None:
        _ANSWER_SYNONYMS = load_answer_synonyms()
    return _ANSWER_SYNONYMS


def normalize_plural_suffix(token):
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") or token.endswith("xes") or token.endswith("zes"):
        return token[:-2]
    if token.endswith("ches") or token.endswith("shes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_safe_plurals(text):
    return " ".join(normalize_plural_suffix(token) for token in text.split())


def normalize_hyphen_slash_spacing(text):
    text = re.sub(r"\s*[-/]\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_basic_answer(text):
    text = "" if text is None else str(text)
    text = clean_markup(text).lower().strip()
    text = normalize_hyphen_slash_spacing(text)
    text = text.translate(_PUNCT_EXCEPT_HYPHEN_SLASH_TABLE)
    text = normalize_hyphen_slash_spacing(text)
    text = _ARTICLES_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_safe_plurals(text)
    return re.sub(r"\s+", " ", text).strip()


def apply_answer_synonyms(text):
    synonyms = answer_synonyms()
    if not synonyms:
        return text
    if text in synonyms:
        return synonyms[text]
    tokens = text.split()
    rewritten = []
    for token in tokens:
        rewritten.append(synonyms.get(token, token))
    return " ".join(rewritten)


def first_answer_span(text):
    text = "" if text is None else str(text)
    text = text.strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0]
    return first_sentence.strip()


def strip_open_prefixes(text):
    text = "" if text is None else str(text)
    text = text.strip()
    changed = True
    while changed:
        changed = False
        lowered = text.lower().lstrip()
        for prefix in _OPEN_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(text) - len(lowered) + len(prefix):].lstrip(" :-,.\t")
                changed = True
                break
    return text.strip()


def clean_markup(text):
    text = "" if text is None else str(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = text.replace("`", "")
    return text.strip()


def strip_candidate_marker(line):
    line = "" if line is None else str(line).strip()
    match = _CANDIDATE_LINE_RE.match(line)
    if match:
        return match.group(1).strip()
    return line


def first_candidate_line(text):
    text = clean_markup(text)
    lines = [strip_candidate_marker(line) for line in text.splitlines()]
    lines = [line.strip(" \t-*:;") for line in lines if line.strip(" \t-*:;")]
    lines = [line for line in lines if line.lower() not in {"answer", "final answer"}]
    if len(lines) <= 1:
        return lines[0] if lines else text

    short_like = []
    for line in lines:
        cleaned = strip_open_prefixes(line).strip()
        words = cleaned.split()
        is_sentence = bool(re.search(r"[.!?]\s+\w", cleaned))
        short_like.append(0 < len(words) <= 12 and not is_sentence)
    if all(short_like):
        return strip_open_prefixes(lines[0]).strip()
    return lines[0]


def choose_closed_answer(text):
    text = "" if text is None else str(text)
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0]
    match = _YES_NO_RE.search(first_sentence.lower())
    if not match:
        match = _YES_NO_RE.search(text.lower())
    if not match:
        return "unknown"
    value = match.group(1).lower()
    if value in {"true", "y", "yeah", "yep"}:
        return "yes"
    if value in {"false", "n", "nope"}:
        return "no"
    return value


def short_open_answer_span(text):
    text = first_candidate_line(text)
    text = clean_markup(text)
    text = strip_open_prefixes(text)
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    first_line = re.sub(r"^(answer|final answer)\s*:\s*", "", first_line, flags=re.IGNORECASE).strip()
    first_line = re.sub(r"^the answer is\s+", "", first_line, flags=re.IGNORECASE).strip()
    colon_match = re.search(r":\s*([^:]+)$", first_line)
    if colon_match:
        candidate = colon_match.group(1).strip()
        if 0 < len(candidate.split()) <= 12:
            first_line = candidate
    for phrase in _OPEN_LEADING_PHRASES:
        pattern = rf"^{re.escape(phrase)}\b\s*"
        first_line = re.sub(pattern, "", first_line, flags=re.IGNORECASE).strip()
    first_line = first_line.rstrip(" .")
    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0].strip()
    if len(first_sentence.split()) <= 12:
        return first_sentence
    for separator in (" because ", " which ", " showing ", " demonstrating ", " suggesting "):
        lowered = first_line.lower()
        idx = lowered.find(separator)
        if idx > 0:
            return first_line[:idx].strip()
    return " ".join(first_sentence.split()[:12]).strip()


def normalize_answer(text, yes_no=False):
    if yes_no:
        return choose_closed_answer(text)

    text = short_open_answer_span(text)
    text = strip_open_prefixes(text)
    text = normalize_basic_answer(text)
    text = apply_answer_synonyms(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_overlap(text):
    text = "" if text is None else str(text)
    text = clean_markup(text).lower()
    text = normalize_basic_answer(text)
    text = apply_answer_synonyms(text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction, answer):
    pred_tokens = prediction.split()
    answer_tokens = answer.split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def clamp01(value):
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def token_precision_recall_f1(prediction, answer):
    pred_tokens = prediction.split()
    answer_tokens = answer.split()
    if not pred_tokens and not answer_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def contains_match(prediction, answer):
    if not prediction or not answer:
        return False
    return prediction in answer or answer in prediction


def char_similarity(prediction, answer):
    if not prediction and not answer:
        return 1.0
    if not prediction or not answer:
        return 0.0
    return SequenceMatcher(None, prediction, answer).ratio()


def word_similarity(prediction, answer):
    pred_tokens = prediction.split()
    answer_tokens = answer.split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0
    return SequenceMatcher(None, pred_tokens, answer_tokens).ratio()


def is_overlong_open_prediction(prediction_norm, answer_norm, max_tokens=20):
    pred_tokens = prediction_norm.split()
    answer_tokens = answer_norm.split()
    if len(pred_tokens) <= max_tokens:
        return False
    if answer_tokens and len(pred_tokens) <= 4 * max(1, len(answer_tokens)):
        return False
    return True


def evaluate_open_match(row, config=None, bertscore_f1=None):
    config = generative_config(**(config or {}))
    prediction = row.get("prediction_norm", "")
    answer = row.get("answer_norm", "")
    empty = not prediction.strip()
    exact = as_bool(row.get("exact_correct")) or prediction == answer
    relaxed = as_bool(row.get("relaxed_correct")) and not empty
    contains = contains_match(prediction, answer)
    precision, recall, f1 = token_precision_recall_f1(prediction, answer)
    c_similarity = char_similarity(prediction, answer)
    w_similarity = word_similarity(prediction, answer)
    bert = clamp01(bertscore_f1)

    overlong = is_overlong_open_prediction(
        prediction,
        answer,
        max_tokens=int(config["overlong_prediction_tokens"]),
    )
    bertscore_correct = bert is not None and bert >= config["hybrid_bertscore_threshold"]
    token_f1_correct = f1 >= config["hybrid_token_f1_threshold"]
    char_similarity_correct = c_similarity >= config["hybrid_char_similarity_threshold"]
    partial = (
        contains
        or f1 >= config["partial_token_f1_threshold"]
        or c_similarity >= config["partial_char_similarity_threshold"]
        or w_similarity >= config["partial_word_similarity_threshold"]
    )
    hybrid = bool(
        exact
        or relaxed
        or contains
        or token_f1_correct
        or char_similarity_correct
        or bertscore_correct
    )

    # Soft score is intentionally continuous rather than 0/1.
    # Exact matches receive 1.0. Relaxed/contains matches receive at least
    # `soft_floor` because they are useful short-answer matches even when
    # wording is not identical. Otherwise, use the strongest available
    # partial/semantic signal across token F1, character similarity, word
    # similarity, and optional per-sample BERTScore.
    component_scores = [f1, c_similarity, w_similarity]
    if bert is not None:
        component_scores.append(bert)
    if exact:
        soft = 1.0
    elif relaxed or contains:
        soft = max(float(config["soft_floor"]), max(component_scores))
    else:
        soft = max(component_scores)

    if empty:
        match_type = "empty_prediction"
    elif exact:
        match_type = "exact"
    elif relaxed:
        match_type = "relaxed"
    elif contains:
        match_type = "contains"
    elif token_f1_correct:
        match_type = "token_f1"
    elif char_similarity_correct:
        match_type = "char_similarity"
    elif bertscore_correct:
        match_type = "bertscore"
    elif overlong:
        match_type = "overlong_prediction"
    elif partial:
        match_type = "partial"
    else:
        match_type = "wrong"

    return {
        "match_type": match_type,
        "contains_match": contains,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "char_similarity": c_similarity,
        "word_similarity": w_similarity,
        "bertscore_f1": bert,
        "partial_match": partial,
        "hybrid_correct": hybrid,
        "soft_score": soft,
        "overlong_prediction": overlong,
        "empty_prediction": empty,
    }


def repeats_question(prediction_norm, question):
    question_norm = normalize_for_overlap(question)
    prediction_norm = normalize_for_overlap(prediction_norm)
    if not prediction_norm or not question_norm:
        return False
    if len(prediction_norm.split()) >= 6 and prediction_norm in question_norm:
        return True
    return token_f1(prediction_norm, question_norm) >= 0.75 and len(prediction_norm.split()) >= 6


def is_descriptive_overlong(prediction_norm, answer_norm):
    pred_tokens = prediction_norm.split()
    answer_tokens = answer_norm.split()
    if not pred_tokens or not answer_tokens:
        return False
    if len(pred_tokens) <= 3 * max(1, len(answer_tokens)):
        return False
    return bool(_DESCRIPTION_WORDS & set(pred_tokens))


def relaxed_correct(prediction_norm, answer_norm, question=None):
    if prediction_norm == answer_norm:
        return True
    if not prediction_norm or not answer_norm:
        return False
    if question and repeats_question(prediction_norm, question):
        return False
    if is_descriptive_overlong(prediction_norm, answer_norm):
        return False
    if prediction_norm in answer_norm or answer_norm in prediction_norm:
        return True
    return token_f1(prediction_norm, answer_norm) >= 0.6


def sentence_bleu(prediction, answer, max_n=4):
    pred_tokens = prediction.split()
    ref_tokens = answer.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append((overlap + 1) / (total + 1) if total else 1.0)

    brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)


def wbss(prediction, answer):
    pred_counts = Counter(prediction.split())
    answer_counts = Counter(answer.split())
    if not pred_counts or not answer_counts:
        return 0.0
    overlap = sum(pred_counts[token] * answer_counts[token] for token in pred_counts.keys() & answer_counts.keys())
    pred_norm = math.sqrt(sum(value * value for value in pred_counts.values()))
    answer_norm = math.sqrt(sum(value * value for value in answer_counts.values()))
    return overlap / (pred_norm * answer_norm) if pred_norm and answer_norm else 0.0


def concept_tokens(text):
    tokens = [
        token for token in text.split()
        if len(token) > 2 and token not in _STOPWORDS and not token.isdigit()
    ]
    return tokens


def cbss(prediction, answer):
    pred_tokens = concept_tokens(prediction)
    answer_tokens = concept_tokens(answer)
    if not pred_tokens or not answer_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def optional_bertscore_scores(open_rows):
    if not open_rows:
        return [], None
    try:
        from bert_score import score
    except ModuleNotFoundError as exc:
        return None, f"bert_score is not installed: {exc}"

    predictions = [row.get("prediction_norm", "") for row in open_rows]
    answers = [row.get("answer_norm", "") for row in open_rows]
    try:
        _, _, f1 = score(predictions, answers, lang="en", verbose=False, rescale_with_baseline=True)
    except Exception as exc:
        return None, str(exc)
    return [float(value) for value in f1.tolist()], None


def optional_bertscore(open_rows):
    scores, error = optional_bertscore_scores(open_rows)
    if scores is None:
        return None, error
    if not scores:
        return None, error
    return float(sum(scores) / len(scores)), error


def compute_metrics(rows, generative_thresholds=None):
    generative_thresholds = generative_config(**(generative_thresholds or {}))
    closed = [row for row in rows if row.get("answer_type") == "closed"]
    open_rows = [row for row in rows if row.get("answer_type") == "open"]

    closed_exact = sum(as_bool(row.get("exact_correct")) for row in closed)
    open_exact = sum(as_bool(row.get("exact_correct")) for row in open_rows)
    open_relaxed = sum(as_bool(row.get("relaxed_correct")) for row in open_rows)
    total_exact = sum(as_bool(row.get("exact_correct")) for row in rows)
    total_relaxed = sum(as_bool(row.get("relaxed_correct")) for row in rows)
    bleu_scores = [sentence_bleu(row.get("prediction_norm", ""), row.get("answer_norm", "")) for row in open_rows]
    wbss_scores = [wbss(row.get("prediction_norm", ""), row.get("answer_norm", "")) for row in open_rows]
    cbss_scores = [cbss(row.get("prediction_norm", ""), row.get("answer_norm", "")) for row in open_rows]
    bertscore_scores, bertscore_error = optional_bertscore_scores(open_rows)
    if bertscore_scores is None:
        bertscore_scores = [None] * len(open_rows)
        bertscore_f1 = None
    else:
        bertscore_f1 = safe_div(sum(bertscore_scores), len(bertscore_scores))
    open_match_details = [
        evaluate_open_match(row, config=generative_thresholds, bertscore_f1=bertscore)
        for row, bertscore in zip(open_rows, bertscore_scores)
    ]
    open_token_precision_scores = [detail["token_precision"] for detail in open_match_details]
    open_token_recall_scores = [detail["token_recall"] for detail in open_match_details]
    open_token_f1_scores = [detail["token_f1"] for detail in open_match_details]
    open_contains = [detail["contains_match"] for detail in open_match_details]
    open_char_scores = [detail["char_similarity"] for detail in open_match_details]
    open_word_scores = [detail["word_similarity"] for detail in open_match_details]
    open_partial = [detail["partial_match"] for detail in open_match_details]
    open_soft = [detail["soft_score"] for detail in open_match_details]
    open_hybrid = [detail["hybrid_correct"] for detail in open_match_details]
    open_generative_scores = [
        (detail["token_f1"] + detail["soft_score"] + float(detail["hybrid_correct"])) / 3
        for detail in open_match_details
    ]

    metrics = {
        "closed_accuracy": safe_div(closed_exact, len(closed)),
        "open_exact_accuracy": safe_div(open_exact, len(open_rows)),
        "open_relaxed_accuracy": safe_div(open_relaxed, len(open_rows)),
        "open_bleu": safe_div(sum(bleu_scores), len(bleu_scores)),
        "open_wbss": safe_div(sum(wbss_scores), len(wbss_scores)),
        "open_bertscore_f1": bertscore_f1,
        "open_cbss": safe_div(sum(cbss_scores), len(cbss_scores)),
        "overall_exact_accuracy": safe_div(total_exact, len(rows)),
        "overall_relaxed_accuracy": safe_div(total_relaxed, len(rows)),
        "closed_samples": len(closed),
        "open_samples": len(open_rows),
        "total_samples": len(rows),
    }
    metrics.update(
        {
            "open_token_precision": safe_div(sum(open_token_precision_scores), len(open_token_precision_scores)),
            "open_token_recall": safe_div(sum(open_token_recall_scores), len(open_token_recall_scores)),
            "open_token_f1": safe_div(sum(open_token_f1_scores), len(open_token_f1_scores)),
            "open_contains_accuracy": safe_div(sum(open_contains), len(open_contains)),
            "open_char_similarity": safe_div(sum(open_char_scores), len(open_char_scores)),
            "open_word_similarity": safe_div(sum(open_word_scores), len(open_word_scores)),
            "open_partial_accuracy": safe_div(sum(open_partial), len(open_partial)),
            "open_soft_score": safe_div(sum(open_soft), len(open_soft)),
            "open_hybrid_accuracy": safe_div(sum(open_hybrid), len(open_hybrid)),
            "overall_token_f1_score": safe_div(closed_exact + sum(open_token_f1_scores), len(rows)),
            "overall_soft_score": safe_div(closed_exact + sum(open_soft), len(rows)),
            "overall_hybrid_accuracy": safe_div(closed_exact + sum(open_hybrid), len(rows)),
            "overall_generative_score": safe_div(closed_exact + sum(open_generative_scores), len(rows)),
        }
    )
    if bertscore_error:
        metrics["open_bertscore_error"] = bertscore_error
    return metrics


def build_error_analysis(rows, generative_thresholds=None):
    generative_thresholds = generative_config(**(generative_thresholds or {}))
    open_rows = [row for row in rows if row.get("answer_type") == "open"]
    bertscore_scores, bertscore_error = optional_bertscore_scores(open_rows)
    if bertscore_scores is None:
        bertscore_scores = [None] * len(open_rows)
    bertscore_by_id = {
        row.get("sample_id", str(index)): score
        for index, (row, score) in enumerate(zip(open_rows, bertscore_scores))
    }

    analysis_rows = []
    counts = Counter()
    for index, row in enumerate(rows):
        answer_type = row.get("answer_type", "")
        if answer_type == "open":
            details = evaluate_open_match(
                row,
                config=generative_thresholds,
                bertscore_f1=bertscore_by_id.get(row.get("sample_id", str(index))),
            )
            match_type = details["match_type"]
        else:
            empty = not str(row.get("prediction_norm", "")).strip()
            details = {
                "contains_match": False,
                "token_precision": "",
                "token_recall": "",
                "token_f1": "",
                "char_similarity": "",
                "word_similarity": "",
                "bertscore_f1": "",
                "partial_match": False,
                "hybrid_correct": as_bool(row.get("exact_correct")),
                "soft_score": 1.0 if as_bool(row.get("exact_correct")) else 0.0,
                "overlong_prediction": False,
                "empty_prediction": empty,
            }
            match_type = "empty_prediction" if empty else ("exact" if as_bool(row.get("exact_correct")) else "wrong")

        counts[match_type] += 1
        analysis_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "answer_type": answer_type,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "prediction_raw": row.get("prediction_raw", ""),
                "prediction_norm": row.get("prediction_norm", ""),
                "answer_norm": row.get("answer_norm", ""),
                "exact_correct": as_bool(row.get("exact_correct")),
                "relaxed_correct": as_bool(row.get("relaxed_correct")),
                "match_type": match_type,
                "contains_match": details["contains_match"],
                "token_precision": details["token_precision"],
                "token_recall": details["token_recall"],
                "token_f1": details["token_f1"],
                "char_similarity": details["char_similarity"],
                "word_similarity": details["word_similarity"],
                "bertscore_f1": details["bertscore_f1"],
                "partial_match": details["partial_match"],
                "hybrid_correct": details["hybrid_correct"],
                "soft_score": details["soft_score"],
                "overlong_prediction": details["overlong_prediction"],
            }
        )

    total = len(analysis_rows)
    summary = {
        "total_samples": total,
        "match_type_counts": dict(counts),
        "match_type_ratios": {key: safe_div(value, total) for key, value in counts.items()},
        "thresholds": generative_thresholds,
    }
    if bertscore_error:
        summary["open_bertscore_error"] = bertscore_error
    return analysis_rows, summary


def save_error_analysis(rows, output_dir, generative_thresholds=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_rows, summary = build_error_analysis(rows, generative_thresholds=generative_thresholds)
    csv_path = output_dir / "error_analysis.csv"
    json_path = output_dir / "error_analysis.json"
    fieldnames = [
        "sample_id",
        "answer_type",
        "question",
        "answer",
        "prediction_raw",
        "prediction_norm",
        "answer_norm",
        "exact_correct",
        "relaxed_correct",
        "match_type",
        "contains_match",
        "token_precision",
        "token_recall",
        "token_f1",
        "char_similarity",
        "word_similarity",
        "bertscore_f1",
        "partial_match",
        "hybrid_correct",
        "soft_score",
        "overlong_prediction",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def save_metrics(metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = output_dir / "metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
