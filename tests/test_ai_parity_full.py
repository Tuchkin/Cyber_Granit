# -*- coding: utf-8 -*-
"""
Расширенная сверка Python- и JavaScript-реализаций ИИ-движка на ВСЕХ данных проекта.

В отличие от test_ai_parity.py (контрольный набор из нескольких примеров), здесь
сравниваются ответы обеих реализаций на:
  - всех 251 сообщениях обучающего датасета и 1280 сообщениях «вне шаблонов»
    (комбинации обращений, сюжетов и окончаний, которых нет в шаблонах датасета);
  - всех 104 описаниях публикаций (OSINT-скоринг);
  - всех 2660 паролях корпуса слабых паролей;
  - всех 34 вопросах базы знаний ИИ-консультанта.

Проверяются: класс и вероятности ансамбля, флаг согласия моделей, подсветка
ключевых слов (сравнивается видимый текст, без учёта способа записи HTML-сущностей),
оценка предсказуемости пароля, ответ консультанта.

Запуск: python tests/test_ai_parity_full.py   (требует установленный Node.js)
"""
import html
import itertools
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "ai_models"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets as ds  # noqa: E402
import nb_engine as eng  # noqa: E402
from test_ai_parity import build_js_harness, extract_script, load_json, run_js  # noqa: E402

GREETINGS = ["", "Ребята, ", "Привет! ", "Внимание! ", "Боец, ", "Друзья, ", "Срочно! ", "Говорят, "]
PLOTS = ["сборы в субботу отменили", "командир просит скинуть фото пропуска", "в штабе проверка документов",
         "клуб переезжает в другое здание", "набор в клуб закрывают", "проголосуй за наш клуб по ссылке",
         "пришли фото плаца для поста", "подпишись на новый канал клуба", "заполни анкету для поездки на сборы",
         "смени пароль, аккаунт пытались взломать", "скинь список отряда",
         "переведи 20 рублей на подарок командиру", "расписание дежурств изменилось",
         "завтра тренировка отменяется", "в клубе утечка данных, срочно смените пароли",
         "сфоткай расписание на стенде", "у командира новый номер", "пришли код, который придёт по СМС",
         "есть подработка на выходные", "сбор у КПП в 9:00"]
ENDINGS = ["", ", передайте всем.", ", это правда?", " до вечера.", ", подробности в личке.",
           ", пока не удалили!", ", так сказали в штабе.", ", ссылка в описании."]


def out_of_template_messages():
    out = []
    for g, c, t in itertools.product(GREETINGS, PLOTS, ENDINGS):
        s = (g + (c if g else c[:1].upper() + c[1:]) + t).strip()
        if not s.endswith((".", "!", "?")):
            s += "!" if g in ("Срочно! ", "Внимание! ") else "."
        out.append(s)
    return out


def main():
    msg_nb, msg_lr = load_json("message_classifier.json"), load_json("message_classifier_logreg.json")
    os_nb, os_lr = load_json("osint_risk_classifier.json"), load_json("osint_risk_classifier_logreg.json")
    pwd = load_json("password_ngram.json")

    messages = [t for t, _ in ds.build_message_dataset(per_class=55)] + out_of_template_messages()
    osint = [t for t, _ in ds.build_osint_dataset(per_class=45)]
    passwords = list(ds.build_weak_password_corpus())
    chat_queries = [(item["q"], True) for item in ds.CHAT_KB]

    payload = {"messages": messages, "osint": osint, "passwords": passwords, "chatQueries": chat_queries}
    js = json.loads(run_js(build_js_harness(extract_script(os.path.join(BASE, "Cyber_Granit.html")),
                                            payload)).strip().splitlines()[-1])

    errors, max_diff = [], 0.0
    for i, t in enumerate(messages):
        py = eng.predict_ensemble(msg_nb, msg_lr, t)
        j = js["ensemble_messages"][i]
        if py["top_class"] != j["topClass"] or py["agree"] != j["agree"]:
            errors.append(f"message[{i}] class/agree mismatch")
        for c in msg_nb["classes"]:
            max_diff = max(max_diff, abs(py["probs"][c] - j["probs"][c]))
        if html.unescape(eng.highlight_words(t, py["top_features"])) != html.unescape(js["highlights"][i]):
            errors.append(f"message[{i}] highlight mismatch")
    if max_diff > 1e-9:
        errors.append(f"probability mismatch: max |dp| = {max_diff:.2e}")
    for i, t in enumerate(osint):
        if eng.predict_ensemble(os_nb, os_lr, t)["top_class"] != js["ensemble_osint"][i]["topClass"]:
            errors.append(f"osint[{i}] mismatch")
    for i, p in enumerate(passwords):
        if abs(eng.score_password_predictability(pwd, p) - js["passwords"][i]) > 0.05:
            errors.append(f"password[{i}] mismatch")
    index = eng.build_tfidf(ds.CHAT_KB)
    for i, (q, _) in enumerate(chat_queries):
        py = eng.chat_answer(ds.CHAT_KB, index, q)
        if py["matched"] != js["chat"][i]["matched"] or py.get("q") != js["chat"][i].get("q"):
            errors.append(f"chat[{i}] mismatch")

    if errors:
        print("FULL PARITY TEST FAILED:")
        for e in errors[:50]:
            print(" -", e)
        sys.exit(1)
    print(f"Full parity OK: {len(messages)} messages, {len(osint)} osint-descriptions, "
          f"{len(passwords)} passwords, {len(chat_queries)} chat queries; "
          f"max |dp| = {max_diff:.1e}")


if __name__ == "__main__":
    main()
