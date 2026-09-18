# -*- coding: utf-8 -*-
"""
Telegram-бот «Кибер-Гранит.ИИ» — ОНЛАЙН-режим проекта.

Это дополнительный, необязательный канал: основной полигон (Cyber_Granit.html
и app.py) работает полностью автономно, без интернета. Бот использует ТУ ЖЕ
самую логику ИИ-моделей (ai_models/nb_engine.py, те же обученные веса), что
и офлайн-версии — просто через интерфейс Telegram, доступный с телефона по
QR-коду. Работа бота требует интернета (так устроен Telegram Bot API), но это
не влияет на офлайн-режим — прохождение полигона не зависит от бота.

Запуск:
    pip install -r requirements.txt
    set TELEGRAM_BOT_TOKEN=<токен от @BotFather>      (Windows PowerShell: $env:TELEGRAM_BOT_TOKEN="...")
    python bot.py
"""
import json
import logging
import os
import sys

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_MODELS_DIR = os.path.join(BASE_DIR, "ai_models")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai_models import nb_engine as eng  # noqa: E402
from ai_models.datasets import CHAT_KB  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cyber-granit-bot")


def _load_json(name):
    with open(os.path.join(AI_MODELS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


MSG_MODEL = _load_json("message_classifier.json")
MSG_LOGREG = _load_json("message_classifier_logreg.json")
OSINT_MODEL = _load_json("osint_risk_classifier.json")
OSINT_LOGREG = _load_json("osint_risk_classifier_logreg.json")
PWD_MODEL = _load_json("password_ngram.json")
CHAT_INDEX = eng.build_tfidf(CHAT_KB)

MSG_LABELS = {
    "phishing": "🎣 ФИШИНГ",
    "social_engineering": "🕵️ ВЕРБОВКА / СОЦИНЖЕНЕРИЯ",
    "ipso_fake": "📢 ИПсО / ФЕЙК",
    "safe": "✅ БЕЗОПАСНО",
}
OSINT_LABELS = {"leak_risk": "🚨 РИСК УТЕЧКИ", "safe": "✅ БЕЗОПАСНО"}

WELCOME_TEXT = (
    "🛡️ *Кибер-Гранит.ИИ* — онлайн-режим\n\n"
    "Это тот же ИИ-модуль, что и в офлайн-версии портала (наивный байес + "
    "логистическая регрессия в ансамбле, символьная языковая модель для паролей, "
    "TF-IDF консультант по базе знаний) — просто доступен прямо в Telegram.\n\n"
    "⚠️ В отличие от офлайн-версии, сообщения в этом чате сохраняются в истории "
    "переписки Telegram — *не присылайте боту реальные пароли или личные данные*, "
    "только тестовые примеры.\n\n"
    "Выберите режим:"
)

MODE_PROMPTS = {
    "check": "🔍 Пришлите текст сообщения (письмо, СМС, пост) — оценю риск фишинга, вербовки, ИПсО.",
    "osint": "👁️ Опишите словами фото или пост, который планируете опубликовать — оценю риск утечки.",
    "password": "🔐 Пришлите *тестовый* пароль (НЕ настоящий!) — оценю его предсказуемость.",
    "ask": "💬 Задайте вопрос по кибербезопасности, законодательству РБ, ОСИНТ или паролям.",
}

MODE_BUTTONS = [
    [InlineKeyboardButton("🔍 Проверить сообщение", callback_data="check")],
    [InlineKeyboardButton("👁️ Оценить риск OSINT", callback_data="osint")],
    [InlineKeyboardButton("🔐 Оценить пароль", callback_data="password")],
    [InlineKeyboardButton("💬 Задать вопрос", callback_data="ask")],
]


def format_ensemble_bars(probs, labels, order):
    lines = []
    for c in order:
        pct = round(probs.get(c, 0.0) * 100, 1)
        bar_len = int(pct / 10)
        bar = "▓" * bar_len + "░" * (10 - bar_len)
        lines.append(f"{labels[c]}: {bar} {pct}%")
    return "\n".join(lines)


def format_agreement(res):
    if res["agree"]:
        return f"🤝 Обе модели (наивный байес и логрег) согласны: {res['nb']['top_class']}."
    return (
        f"⚠️ Модели разошлись: наивный байес → {res['nb']['top_class']}, "
        f"логрег → {res['logreg']['top_class']}. Итог — среднее вероятностей обеих."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = None
    await update.message.reply_text(
        WELCOME_TEXT, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(MODE_BUTTONS)
    )


async def menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode = query.data
    context.user_data["mode"] = mode
    await query.message.reply_text(MODE_PROMPTS[mode], parse_mode="Markdown")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = context.user_data.get("mode")
    text = update.message.text or ""

    if mode == "check":
        res = eng.predict_ensemble(MSG_MODEL, MSG_LOGREG, text)
        top = res["top_class"]
        conf = round(res["probs"][top] * 100, 1)
        reply = (
            f"*{MSG_LABELS[top]}* (уверенность {conf}%)\n\n"
            + format_ensemble_bars(res["probs"], MSG_LABELS, ["phishing", "social_engineering", "ipso_fake", "safe"])
            + "\n\n"
        )
        if res["top_features"]:
            reply += f"Ключевые слова: {', '.join(res['top_features'])}\n\n"
        reply += format_agreement(res)
        await update.message.reply_text(reply, parse_mode="Markdown")

    elif mode == "osint":
        res = eng.predict_ensemble(OSINT_MODEL, OSINT_LOGREG, text)
        top = res["top_class"]
        reply = (
            f"*{OSINT_LABELS[top]}*\n\n"
            + format_ensemble_bars(res["probs"], OSINT_LABELS, ["leak_risk", "safe"])
            + "\n\n" + format_agreement(res)
        )
        await update.message.reply_text(reply, parse_mode="Markdown")

    elif mode == "password":
        pct = eng.score_password_predictability(PWD_MODEL, text)
        if pct >= 65:
            verdict = "⚠️ Похож на распространённый шаблон — ИИ-модель считает его предсказуемым."
        elif pct >= 30:
            verdict = "🟡 Есть отдельные узнаваемые фрагменты."
        else:
            verdict = "✅ Структура символов статистически близка к случайной."
        await update.message.reply_text(
            f"🔐 AI-оценка предсказуемости: *{pct}%*\n{verdict}\n\n"
            "_Напоминание: это учебный тренажёр, не отправляйте сюда реальные пароли._",
            parse_mode="Markdown",
        )

    else:
        res = eng.chat_answer(CHAT_KB, CHAT_INDEX, text)
        if res.get("matched"):
            await update.message.reply_text(
                f"🔎 _Похожий вопрос в базе: «{res['q']}»_\n\n{res['text']}", parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(res["text"])


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print(
            "Не найден токен. Установите переменную окружения TELEGRAM_BOT_TOKEN "
            "(получить токен: Telegram -> @BotFather -> /newbot)."
        )
        return

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(menu_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Кибер-Гранит.ИИ бот запущен (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
