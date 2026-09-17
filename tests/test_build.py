# -*- coding: utf-8 -*-
"""
Дымовой тест: обучение ИИ-моделей должно проходить без ошибок и давать
структурно корректный результат с приемлемой точностью на отложенной выборке.

Запуск: python tests/test_build.py
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_DIR = os.path.join(BASE, "ai_models")

MIN_ACCURACY = 0.85


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "build_models.py"], cwd=AI_DIR, capture_output=True,
        text=True, encoding="utf-8", env=env,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)

    lines = [l for l in result.stdout.splitlines() if "accuracy:" in l]
    for line in lines:
        pct = float(line.split(":")[1].strip().rstrip("%"))
        assert pct / 100.0 >= MIN_ACCURACY, f"Accuracy too low: {line}"

    with open(os.path.join(AI_DIR, "message_classifier.json"), encoding="utf-8") as f:
        msg = json.load(f)
    with open(os.path.join(AI_DIR, "osint_risk_classifier.json"), encoding="utf-8") as f:
        osint = json.load(f)
    with open(os.path.join(AI_DIR, "password_ngram.json"), encoding="utf-8") as f:
        pwd = json.load(f)

    assert msg["type"] == "nb_char_ngram"
    assert set(msg["classes"]) == {"phishing", "social_engineering", "ipso_fake", "safe"}
    assert osint["type"] == "nb_char_ngram"
    assert set(osint["classes"]) == {"leak_risk", "safe"}
    assert pwd["type"] == "char_lm"
    assert "calibration" in pwd

    print("Build + structural checks OK")


if __name__ == "__main__":
    main()
