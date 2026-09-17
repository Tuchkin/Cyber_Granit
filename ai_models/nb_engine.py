# -*- coding: utf-8 -*-
"""
Кибер-Гранит.ИИ — общий движок лёгких офлайн-моделей.

Два алгоритма, оба реализованы "с нуля" на чистом Python (без sklearn/numpy),
чтобы их было легко один-в-один портировать на JavaScript для автономной
HTML-версии (Cyber_Granit.html) и использовать напрямую в Streamlit-версии
(app.py). Никаких сетевых вызовов, никаких внешних API — все модели
обучаются заранее (build_models.py) и грузятся из JSON-файлов.

1) Наивный байесовский классификатор текста на символьных n-граммах слов
   (устойчив к русской словоформе без стемминга/лемматизации).
   Используется для:
     - классификатора угроз в сообщениях (фишинг/вербовка/ИПсО/безопасно)
     - скоринга риска OSINT-публикации (риск утечки / безопасно)

2) Символьная n-граммная языковая модель (марковская цепь символов)
   для оценки "типичности" пароля относительно корпуса известных слабых
   паролей-паттернов — лёгкая, но настоящая статистическая ML-модель,
   а не просто regex.
"""
import re
import math
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Токенизация
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[^0-9a-zа-яё\s]", re.IGNORECASE)


def normalize(text):
    text = (text or "").lower()
    text = _WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_tokens(text):
    norm = normalize(text)
    return [w for w in norm.split(" ") if w]


def char_ngrams_of_word(word, n=3):
    pad = "_" * (n - 1)
    padded = pad + word + pad
    if len(padded) < n:
        return [padded]
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def char_ngrams(text, n=3):
    grams = []
    for w in word_tokens(text):
        grams.extend(char_ngrams_of_word(w, n))
    return grams


# ---------------------------------------------------------------------------
# 1) Наивный байесовский классификатор (Multinomial NB, char n-grams)
# ---------------------------------------------------------------------------

def train_nb(examples, classes=None, n=3, alpha=0.5):
    """examples: список (text, label)."""
    if classes is None:
        classes = sorted(set(lbl for _, lbl in examples))

    class_docs = {c: [] for c in classes}
    for text, lbl in examples:
        if lbl in class_docs:
            class_docs[lbl].append(text)

    vocab_counts = {c: Counter() for c in classes}
    for text, lbl in examples:
        if lbl not in vocab_counts:
            continue
        vocab_counts[lbl].update(char_ngrams(text, n))

    vocab = set()
    for c in classes:
        vocab.update(vocab_counts[c].keys())
    vocab = sorted(vocab)
    vocab_size = len(vocab)

    total_docs = sum(len(v) for v in class_docs.values())
    class_totals = {c: sum(vocab_counts[c].values()) for c in classes}

    log_prior = {c: math.log(max(len(class_docs[c]), 1) / total_docs) for c in classes}

    log_likelihood = {c: {} for c in classes}
    for c in classes:
        denom = class_totals[c] + alpha * vocab_size
        for g, cnt in vocab_counts[c].items():
            log_likelihood[c][g] = math.log((cnt + alpha) / denom)

    default_log_likelihood = {
        c: math.log(alpha / (class_totals[c] + alpha * vocab_size)) for c in classes
    }

    return {
        "type": "nb_char_ngram",
        "classes": classes,
        "n": n,
        "alpha": alpha,
        "vocab_size": vocab_size,
        "log_prior": log_prior,
        "log_likelihood": log_likelihood,
        "default_log_likelihood": default_log_likelihood,
        "train_docs": total_docs,
        "docs_per_class": {c: len(class_docs[c]) for c in classes},
    }


def predict_nb(model, text, top_words=4):
    classes = model["classes"]
    n = model["n"]
    words = word_tokens(text)

    scores = {c: model["log_prior"][c] for c in classes}
    word_scores = {w: {c: 0.0 for c in classes} for w in set(words)}

    for w in words:
        grams = char_ngrams_of_word(w, n)
        for c in classes:
            ll = model["log_likelihood"][c]
            default = model["default_log_likelihood"][c]
            s = sum(ll.get(g, default) for g in grams)
            scores[c] += s
            word_scores[w][c] += s

    m = max(scores.values())
    exps = {c: math.exp(scores[c] - m) for c in classes}
    z = sum(exps.values()) or 1.0
    probs = {c: exps[c] / z for c in classes}

    top_class = max(probs, key=probs.get)
    other_classes = [c for c in classes if c != top_class] or [top_class]

    contrib = []
    for w in set(words):
        c_top = word_scores[w][top_class]
        c_other_avg = sum(word_scores[w][c] for c in other_classes) / len(other_classes)
        contrib.append((w, c_top - c_other_avg))
    contrib.sort(key=lambda x: -x[1])
    top_features = [w for w, v in contrib if v > 0][:top_words]

    return {"probs": probs, "top_class": top_class, "top_features": top_features}


def evaluate_nb(model, examples, n=3):
    classes = model["classes"]
    confusion = {c: {c2: 0 for c2 in classes} for c in classes}
    correct = 0
    for text, true_label in examples:
        pred = predict_nb(model, text)["top_class"]
        confusion[true_label][pred] += 1
        if pred == true_label:
            correct += 1
    accuracy = correct / len(examples) if examples else 0.0

    per_class = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[c2][c] for c2 in classes if c2 != c)
        fn = sum(confusion[c][c2] for c2 in classes if c2 != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}

    return {"accuracy": accuracy, "confusion": confusion, "per_class": per_class}


# ---------------------------------------------------------------------------
# 1b) Многоклассовая логистическая регрессия (softmax) на тех же признаках —
#     второй, принципиально иной алгоритм (дискриминативный, не байесовский),
#     обученный градиентным спуском по разреженным признакам. Используется
#     как независимая вторая модель для ансамбля с наивным байесом.
# ---------------------------------------------------------------------------

def train_logreg(examples, classes=None, n=3, epochs=300, lr=0.5, l2=0.002):
    if classes is None:
        classes = sorted(set(lbl for _, lbl in examples))
    class_idx = {c: i for i, c in enumerate(classes)}
    C = len(classes)

    vocab = {}
    sparse_docs = []
    for text, lbl in examples:
        if lbl not in class_idx:
            continue
        counts = Counter(char_ngrams(text, n))
        feats = {}
        for g, cnt in counts.items():
            if g not in vocab:
                vocab[g] = len(vocab)
            feats[vocab[g]] = cnt
        sparse_docs.append((feats, class_idx[lbl]))

    V = len(vocab)
    W = [[0.0] * V for _ in range(C)]
    b = [0.0] * C
    N = len(sparse_docs) or 1

    for _ in range(epochs):
        grad_W = [dict() for _ in range(C)]
        grad_b = [0.0] * C
        for feats, y in sparse_docs:
            scores = [b[c] + sum(W[c][idx] * val for idx, val in feats.items()) for c in range(C)]
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            z = sum(exps) or 1.0
            probs = [e / z for e in exps]
            for c in range(C):
                diff = probs[c] - (1.0 if c == y else 0.0)
                if diff == 0.0:
                    continue
                grad_b[c] += diff
                gW = grad_W[c]
                for idx, val in feats.items():
                    gW[idx] = gW.get(idx, 0.0) + diff * val
        for c in range(C):
            b[c] -= lr * grad_b[c] / N
            Wc = W[c]
            for idx, g in grad_W[c].items():
                Wc[idx] -= lr * (g / N + l2 * Wc[idx])

    return {
        "type": "logreg_char_ngram",
        "classes": classes,
        "n": n,
        "vocab": vocab,
        "weights": W,
        "bias": b,
        "epochs": epochs,
        "lr": lr,
        "l2": l2,
    }


def predict_logreg(model, text):
    classes = model["classes"]
    C = len(classes)
    counts = Counter(char_ngrams(text, model["n"]))
    vocab = model["vocab"]
    feats = {}
    for g, cnt in counts.items():
        idx = vocab.get(g)
        if idx is not None:
            feats[idx] = feats.get(idx, 0) + cnt

    scores = [model["bias"][c] + sum(model["weights"][c][idx] * val for idx, val in feats.items()) for c in range(C)]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps) or 1.0
    probs = {classes[c]: exps[c] / z for c in range(C)}
    top_class = max(probs, key=probs.get)
    return {"probs": probs, "top_class": top_class}


def evaluate_logreg(model, examples):
    classes = model["classes"]
    confusion = {c: {c2: 0 for c2 in classes} for c in classes}
    correct = 0
    for text, true_label in examples:
        pred = predict_logreg(model, text)["top_class"]
        confusion[true_label][pred] += 1
        if pred == true_label:
            correct += 1
    accuracy = correct / len(examples) if examples else 0.0
    return {"accuracy": accuracy, "confusion": confusion}


def predict_ensemble(nb_model, logreg_model, text):
    """Усредняет вероятности двух независимых моделей (Naive Bayes + LogReg)."""
    nb_res = predict_nb(nb_model, text)
    lr_res = predict_logreg(logreg_model, text)
    classes = nb_model["classes"]
    avg_probs = {c: (nb_res["probs"][c] + lr_res["probs"].get(c, 0.0)) / 2.0 for c in classes}
    top_class = max(avg_probs, key=avg_probs.get)
    agree = nb_res["top_class"] == lr_res["top_class"]
    return {
        "probs": avg_probs,
        "top_class": top_class,
        "nb": nb_res,
        "logreg": lr_res,
        "agree": agree,
        "top_features": nb_res["top_features"],
    }


# ---------------------------------------------------------------------------
# 2) Символьная n-граммная языковая модель (для паролей)
# ---------------------------------------------------------------------------

def train_char_lm(corpus_words, order=3, alpha=0.5):
    context_counts = defaultdict(Counter)
    for w in corpus_words:
        w = w.lower()
        padded = ("_" * order) + w + "_"
        for i in range(len(padded) - order):
            ctx = padded[i:i + order]
            nxt = padded[i + order]
            context_counts[ctx][nxt] += 1

    alphabet = set()
    for counter in context_counts.values():
        alphabet.update(counter.keys())
    alphabet = sorted(alphabet)
    V = max(len(alphabet), 1)

    log_prob = {}
    context_totals = {}
    for ctx, counter in context_counts.items():
        total = sum(counter.values())
        context_totals[ctx] = total
        log_prob[ctx] = {ch: math.log((cnt + alpha) / (total + alpha * V)) for ch, cnt in counter.items()}

    default_lp_known_ctx = {
        ctx: math.log(alpha / (context_totals[ctx] + alpha * V)) for ctx in context_counts
    }
    global_default = math.log(alpha / (alpha * V))

    return {
        "type": "char_lm",
        "order": order,
        "alpha": alpha,
        "vocab_size": V,
        "log_prob": log_prob,
        "default_lp_known_ctx": default_lp_known_ctx,
        "global_default": global_default,
    }


def score_password_lm_raw(model, password):
    order = model["order"]
    padded = ("_" * order) + (password or "").lower() + "_"
    total_lp = 0.0
    count = 0
    for i in range(len(padded) - order):
        ctx = padded[i:i + order]
        nxt = padded[i + order]
        ctx_probs = model["log_prob"].get(ctx)
        if ctx_probs is not None and nxt in ctx_probs:
            lp = ctx_probs[nxt]
        elif ctx in model["default_lp_known_ctx"]:
            lp = model["default_lp_known_ctx"][ctx]
        else:
            lp = model["global_default"]
        total_lp += lp
        count += 1
    return total_lp / count if count else model["global_default"]


def score_password_predictability(model, password):
    """0..100: чем выше — тем пароль ближе к типичным слабым паттернам."""
    if not password:
        return 0.0
    avg_lp = score_password_lm_raw(model, password)
    lo = model["calibration"]["min_lp"]
    hi = model["calibration"]["max_lp"]
    if hi - lo < 1e-9:
        return 0.0
    pct = (avg_lp - lo) / (hi - lo)
    pct = max(0.0, min(1.0, pct))
    return round(pct * 100, 1)


# ---------------------------------------------------------------------------
# 3) ИИ-консультант: TF-IDF + косинусное сходство по базе знаний
# ---------------------------------------------------------------------------

def build_tfidf(kb):
    """kb: список dict с ключами 'q' и 'a'."""
    docs = [word_tokens(item["q"] + " " + item["q"] + " " + item["a"]) for item in kb]
    df = Counter()
    for tokens in docs:
        df.update(set(tokens))
    n_docs = len(docs)
    idf = {t: math.log((n_docs + 1) / (c + 1)) + 1 for t, c in df.items()}
    vectors = []
    for tokens in docs:
        tf = Counter(tokens)
        vectors.append({t: cnt * idf.get(t, 0.0) for t, cnt in tf.items()})
    return {"idf": idf, "vectors": vectors}


def _cosine(vec_a, vec_b):
    dot = sum(v * vec_b.get(k, 0.0) for k, v in vec_a.items())
    na = math.sqrt(sum(v * v for v in vec_a.values()))
    nb = math.sqrt(sum(v * v for v in vec_b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chat_answer(kb, index, question, threshold=0.12):
    tokens = word_tokens(question)
    tf = Counter(tokens)
    q_vec = {t: cnt * index["idf"].get(t, 0.0) for t, cnt in tf.items()}

    best_i, best_score = -1, 0.0
    for i, vec in enumerate(index["vectors"]):
        score = _cosine(q_vec, vec)
        if score > best_score:
            best_score, best_i = score, i

    if best_i == -1 or best_score < threshold:
        return {
            "matched": False,
            "text": "Пока не нашёл точного ответа в базе знаний портала. Попробуйте "
                    "переформулировать вопрос или посмотрите соответствующий раздел в меню слева.",
        }
    return {"matched": True, "q": kb[best_i]["q"], "text": kb[best_i]["a"], "score": best_score}
