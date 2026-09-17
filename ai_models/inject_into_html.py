# -*- coding: utf-8 -*-
"""
Встраивает веса обученных ИИ-моделей (JSON) прямо в Cyber_Granit.html,
чтобы файл оставался единым автономным документом (открывается двойным
кликом, без сервера и без интернета — fetch() локальных JSON из file://
блокируется браузерами по CORS, поэтому веса должны быть буквально внутри
<script> как JS-константы).

Запуск: python inject_into_html.py
Требует, чтобы build_models.py уже был выполнен (message_classifier.json,
osint_risk_classifier.json, password_ngram.json лежат рядом).
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(os.path.dirname(BASE_DIR), "Cyber_Granit.html")
MARKER = "/*__AI_MODELS_INJECT__*/"


def load(name):
    with open(os.path.join(BASE_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    msg_model = load("message_classifier.json")
    osint_model = load("osint_risk_classifier.json")
    pwd_model = load("password_ngram.json")

    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()

    if MARKER not in html:
        raise SystemExit(f"Маркер {MARKER} не найден в {HTML_PATH} — сначала добавьте его в <script>.")

    snippet = (
        "const AI_MSG_MODEL = " + json.dumps(msg_model, ensure_ascii=False) + ";\n"
        "const AI_OSINT_MODEL = " + json.dumps(osint_model, ensure_ascii=False) + ";\n"
        "const AI_PWD_MODEL = " + json.dumps(pwd_model, ensure_ascii=False) + ";"
    )

    html = html.replace(MARKER, snippet)

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Модели встроены в {HTML_PATH} ({len(snippet)/1024:.1f} КБ добавлено).")


if __name__ == "__main__":
    main()
