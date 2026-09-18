# -*- coding: utf-8 -*-
"""
Генерирует QR-код на Telegram-бота для размещения в материалах полигона
(например, на слайде «Онлайн-режим» или на стенде на форуме).

Запуск:
    python generate_qr.py <username_бота_без_@>
    (пример: python generate_qr.py CyberGranitBot)

Результат: telegram_bot/qr-code.png и ../images/telegram-bot-qr.png
(последний — чтобы им можно было сразу пользоваться в Cyber_Granit.html/app.py).
"""
import os
import sys

import qrcode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(os.path.dirname(BASE_DIR), "images")


def main():
    if len(sys.argv) < 2:
        print("Использование: python generate_qr.py <username_бота_без_@>")
        sys.exit(1)

    username = sys.argv[1].lstrip("@")
    url = f"https://t.me/{username}"

    img = qrcode.make(url, box_size=10, border=2)

    local_out = os.path.join(BASE_DIR, "qr-code.png")
    img.save(local_out)
    print(f"Сохранено: {local_out}")

    if os.path.isdir(IMAGES_DIR):
        images_out = os.path.join(IMAGES_DIR, "telegram-bot-qr.png")
        img.save(images_out)
        print(f"Сохранено: {images_out}")

    print(f"QR ведёт на: {url}")


if __name__ == "__main__":
    main()
