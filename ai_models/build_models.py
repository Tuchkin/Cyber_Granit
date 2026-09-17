# -*- coding: utf-8 -*-
"""
Обучение офлайн-ИИ-моделей проекта «Кибер-Гранит.ИИ».

Запуск:  python build_models.py

Ничего не скачивает из интернета — все данные и алгоритмы находятся в
этой папке (datasets.py, nb_engine.py). Скрипт:
  1. генерирует обучающие датасеты (datasets.py);
  2. делит их на train/test (80/20, воспроизводимо, seed=42);
  3. обучает 3 модели (nb_engine.py);
  4. считает метрики качества на отложенной выборке;
  5. сохраняет веса моделей в JSON (для Streamlit-версии и для встраивания
     в автономный Cyber_Granit.html) и отчёт MODEL_CARD.md.
"""
import json
import math
import random
import time
import os

import nb_engine as eng
import datasets as ds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(42)


def train_test_split(examples, test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for text, lbl in examples:
        by_class.setdefault(lbl, []).append((text, lbl))
    train, test = [], []
    for lbl, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        test += items[:n_test]
        train += items[n_test:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def fmt_pct(x):
    return f"{x * 100:.1f}%"


def build_message_classifier(report_lines):
    t0 = time.time()
    data = ds.build_message_dataset(per_class=55)
    train, test = train_test_split(data)
    model = eng.train_nb(train, n=3, alpha=0.5)
    train_time = time.time() - t0

    metrics = eng.evaluate_nb(model, test)

    report_lines.append("## 1. Классификатор угроз в сообщениях (message_classifier)\n")
    report_lines.append(f"- Алгоритм: наивный байесовский классификатор на символьных 3-граммах слов (Laplace-сглаживание, α=0.5)\n")
    report_lines.append(f"- Классы: {', '.join(model['classes'])}\n")
    report_lines.append(f"- Размер датасета: {len(data)} примеров ({len(train)} train / {len(test)} test, 80/20)\n")
    report_lines.append(f"- Размер словаря n-грамм: {model['vocab_size']}\n")
    report_lines.append(f"- Время обучения: {train_time*1000:.1f} мс\n")
    report_lines.append(f"- **Accuracy на отложенной выборке: {fmt_pct(metrics['accuracy'])}**\n")
    report_lines.append("\n| Класс | Precision | Recall | F1 |\n|---|---|---|---|\n")
    for c in model["classes"]:
        pc = metrics["per_class"][c]
        report_lines.append(f"| {c} | {fmt_pct(pc['precision'])} | {fmt_pct(pc['recall'])} | {fmt_pct(pc['f1'])} |\n")
    report_lines.append("\n")

    demo_texts = [
        "Боец, срочно перейди по ссылке и введи пароль для подтверждения",
        "Заплачу 200 рублей, если сфотографируешь КПП и пришлешь координаты",
        "СРОЧНО!!! Всех расформировывают!!! РЕПОСТ пока не удалили!!!",
        "Расписание тренировок на среду без изменений, начало в 17:00",
    ]
    report_lines.append("Примеры работы модели:\n\n")
    for t in demo_texts:
        pred = eng.predict_nb(model, t)
        top = pred["top_class"]
        p = pred["probs"][top]
        report_lines.append(f"- «{t}» → **{top}** ({fmt_pct(p)}), ключевые слова: {', '.join(pred['top_features']) or '—'}\n")
    report_lines.append("\n")

    return model, metrics


def build_osint_classifier(report_lines):
    t0 = time.time()
    data = ds.build_osint_dataset(per_class=45)
    train, test = train_test_split(data)
    model = eng.train_nb(train, n=3, alpha=0.5)
    train_time = time.time() - t0

    metrics = eng.evaluate_nb(model, test)

    report_lines.append("## 2. Скоринг риска OSINT-публикации (osint_risk_classifier)\n")
    report_lines.append("- Алгоритм: тот же наивный байесовский классификатор на символьных 3-граммах\n")
    report_lines.append(f"- Классы: {', '.join(model['classes'])}\n")
    report_lines.append(f"- Размер датасета: {len(data)} примеров ({len(train)} train / {len(test)} test)\n")
    report_lines.append(f"- Время обучения: {train_time*1000:.1f} мс\n")
    report_lines.append(f"- **Accuracy на отложенной выборке: {fmt_pct(metrics['accuracy'])}**\n")
    report_lines.append("\n| Класс | Precision | Recall | F1 |\n|---|---|---|---|\n")
    for c in model["classes"]:
        pc = metrics["per_class"][c]
        report_lines.append(f"| {c} | {fmt_pct(pc['precision'])} | {fmt_pct(pc['recall'])} | {fmt_pct(pc['f1'])} |\n")
    report_lines.append("\n")

    return model, metrics


def build_password_model(report_lines):
    t0 = time.time()
    weak_corpus = ds.build_weak_password_corpus()
    model = eng.train_char_lm(weak_corpus, order=3, alpha=0.5)
    train_time = time.time() - t0

    strong_ref = ds.build_strong_reference_passwords()
    weak_scores = [eng.score_password_lm_raw(model, w) for w in weak_corpus]
    strong_scores = [eng.score_password_lm_raw(model, s) for s in strong_ref]

    max_lp = sorted(weak_scores)[int(len(weak_scores) * 0.95)]
    min_lp = sorted(strong_scores)[int(len(strong_scores) * 0.05)]
    model["calibration"] = {"min_lp": min_lp, "max_lp": max_lp}

    def predictability(pwd):
        return eng.score_password_predictability(model, pwd)

    report_lines.append("## 3. AI-оценка предсказуемости пароля (password_ngram)\n")
    report_lines.append("- Алгоритм: символьная n-граммная языковая модель (order=3, Laplace-сглаживание α=0.5), "
                         "обученная на корпусе типовых слабых паролей-паттернов\n")
    report_lines.append(f"- Размер обучающего корпуса: {len(weak_corpus)} строк, алфавит {model['vocab_size']} символов\n")
    report_lines.append(f"- Время обучения: {train_time*1000:.1f} мс\n")
    report_lines.append("- Калибровка шкалы 0–100%: 95-й перцентиль правдоподобия слабого корпуса (max) "
                         "и 5-й перцентиль правдоподобия случайных 14–22-символьных эталонов (min)\n\n")

    demo_pwds = ["VPK_Granit_2025!", "Zxcvbnm123456!@#", "Bronya_Granit_Rubezh_Sever_99!", "qwerty123", "kH7$mQ2!vLpZ9#eR"]
    report_lines.append("Примеры работы модели (AI-оценка предсказуемости, выше = более типично/слабее):\n\n")
    for p in demo_pwds:
        report_lines.append(f"- `{p}` → {predictability(p)}%\n")
    report_lines.append("\n")

    return model


def main():
    out_dir = BASE_DIR
    report = []
    report.append("# MODEL CARD — «Кибер-Гранит.ИИ»\n\n")
    report.append("Автоматически сгенерированный отчёт об обучении офлайн-моделей. "
                   "Пересоздаётся командой `python build_models.py`.\n\n")
    report.append("Все модели обучаются **локально, без интернета и без сторонних библиотек** "
                   "(чистый Python: `re`, `math`, `collections`), на синтетическом учебном датасете "
                   "(см. `datasets.py`). Это гарантирует полную автономность конечного продукта.\n\n")

    msg_model, msg_metrics = build_message_classifier(report)
    osint_model, osint_metrics = build_osint_classifier(report)
    pwd_model = build_password_model(report)

    with open(os.path.join(out_dir, "message_classifier.json"), "w", encoding="utf-8") as f:
        json.dump(msg_model, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "osint_risk_classifier.json"), "w", encoding="utf-8") as f:
        json.dump(osint_model, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "password_ngram.json"), "w", encoding="utf-8") as f:
        json.dump(pwd_model, f, ensure_ascii=False)

    with open(os.path.join(out_dir, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.writelines(report)

    print("Готово. Файлы сохранены в", out_dir)
    print(" - message_classifier.json")
    print(" - osint_risk_classifier.json")
    print(" - password_ngram.json")
    print(" - MODEL_CARD.md")
    print()
    print(f"message_classifier accuracy: {fmt_pct(msg_metrics['accuracy'])}")
    print(f"osint_risk_classifier accuracy: {fmt_pct(osint_metrics['accuracy'])}")


if __name__ == "__main__":
    main()
