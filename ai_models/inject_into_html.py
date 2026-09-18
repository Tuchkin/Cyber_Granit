# -*- coding: utf-8 -*-
"""
Встраивает веса обученных ИИ-моделей (JSON) прямо в Cyber_Granit.html,
чтобы файл оставался единым автономным документом (открывается двойным
кликом, без сервера и без интернета — fetch() локальных JSON из file://
блокируется браузерами по CORS, поэтому веса должны быть буквально внутри
<script> как JS-константы).

Идемпотентно: ищет пару маркеров /*__AI_MODELS_START__*/ ... /*__AI_MODELS_END__*/
в <script> и полностью заменяет содержимое между ними — можно запускать
многократно при переобучении моделей.

Запуск: python inject_into_html.py
Требует, чтобы build_models.py уже был выполнен.
"""
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(os.path.dirname(BASE_DIR), "Cyber_Granit.html")
START_MARKER = "/*__AI_MODELS_START__*/"
END_MARKER = "/*__AI_MODELS_END__*/"

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from datasets import CHAT_KB  # noqa: E402

MODELS = [
    ("AI_MSG_MODEL", "message_classifier.json"),
    ("AI_MSG_LOGREG_MODEL", "message_classifier_logreg.json"),
    ("AI_OSINT_MODEL", "osint_risk_classifier.json"),
    ("AI_OSINT_LOGREG_MODEL", "osint_risk_classifier_logreg.json"),
    ("AI_PWD_MODEL", "password_ngram.json"),
]


def load(name):
    with open(os.path.join(BASE_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()

    if START_MARKER not in html or END_MARKER not in html:
        raise SystemExit(
            f"Маркеры {START_MARKER} / {END_MARKER} не найдены в {HTML_PATH} — "
            "добавьте их в <script> (пустая пара маркеров подряд)."
        )

    lines = []
    total_kb = 0.0
    for const_name, filename in MODELS:
        model = load(filename)
        dumped = json.dumps(model, ensure_ascii=False)
        total_kb += len(dumped) / 1024
        lines.append(f"const {const_name} = {dumped};")

    # База знаний ИИ-консультанта — единый источник правды datasets.CHAT_KB
    # (раньше была вручную продублирована прямо в HTML — риск рассинхронизации
    # с Python-версией; теперь генерируется отсюда же, что и модели).
    chat_kb_dumped = json.dumps(CHAT_KB, ensure_ascii=False)
    total_kb += len(chat_kb_dumped) / 1024
    lines.append(f"const AI_CHAT_KB = {chat_kb_dumped};")

    snippet = START_MARKER + "\n" + "\n".join(lines) + "\n" + END_MARKER

    pattern = re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER)
    html, count = re.subn(pattern, lambda m: snippet, html, count=1, flags=re.DOTALL)
    if count != 1:
        raise SystemExit("Не удалось заменить блок между маркерами.")

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Встроено {len(MODELS)} моделей + база знаний ({len(CHAT_KB)} Q&A) в {HTML_PATH} ({total_kb:.1f} КБ).")


if __name__ == "__main__":
    main()
