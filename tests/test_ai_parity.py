# -*- coding: utf-8 -*-
"""
Проверка идентичности поведения Python- и JavaScript-реализаций ИИ-движка.

Модели обучаются один раз в Python (ai_models/build_models.py), затем их веса
встраиваются в Cyber_Granit.html (ai_models/inject_into_html.py) и в браузере
используются собственной JS-реализацией того же алгоритма (см. <script> в
Cyber_Granit.html). Этот тест гарантирует, что обе реализации дают одинаковый
результат на контрольных примерах — то есть JS-порт не разошёлся с Python-
оригиналом после очередных правок.

Запуск: python tests/test_ai_parity.py   (требует установленный Node.js)
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "ai_models"))

import nb_engine as eng  # noqa: E402

TEST_MESSAGES = [
    "Боец, срочно перейди по ссылке и введи пароль для подтверждения",
    "Заплачу 200 рублей, если сфотографируешь КПП и пришлешь координаты",
    "СРОЧНО!!! Всех расформировывают!!! РЕПОСТ пока не удалили!!!",
    "Расписание тренировок на среду без изменений, начало в 17:00",
]
TEST_OSINT = [
    "Селфи в штабе, где на заднем плане видна доска с расписанием дежурств и картой района.",
    "Фото в камуфляже на фоне городского памятника с геометкой центра города.",
]
TEST_PASSWORDS = [
    "VPK_Granit_2025!", "Zxcvbnm123456!@#", "Bronya_Granit_Rubezh_Sever_99!",
    "qwerty123", "kH7$mQ2!vLpZ9#eR",
]


def load_json(name):
    with open(os.path.join(BASE, "ai_models", name), encoding="utf-8") as f:
        return json.load(f)


def extract_script(html_path):
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    start = html.index("<script>") + len("<script>")
    end = html.index("</script>", start)
    return html[start:end]


def run_js(js_code):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        ["node", "-"], input=js_code, capture_output=True, text=True,
        encoding="utf-8", env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError("Node.js error:\n" + proc.stderr)
    return proc.stdout


def build_js_harness(script_code, payload):
    return (
        "function stub(){return {value:'',innerHTML:'',style:{},"
        "classList:{add(){},remove(){},contains(){return false;}},"
        "textContent:'',scrollTop:0,scrollHeight:0};}\n"
        "global.document = {getElementById:()=>stub(), querySelectorAll:()=>[], querySelector:()=>stub()};\n"
        "global.window = {scrollTo(){}};\n"
        + script_code
        + "\nconst testData = " + json.dumps(payload, ensure_ascii=False) + ";\n"
        "const out = {messages: [], osint: [], passwords: [], ensemble_messages: [], ensemble_osint: []};\n"
        "testData.messages.forEach(t => { const r = aiPredictNB(AI_MSG_MODEL, t); out.messages.push({topClass: r.topClass, probs: r.probs}); });\n"
        "testData.osint.forEach(t => { const r = aiPredictNB(AI_OSINT_MODEL, t); out.osint.push({topClass: r.topClass, probs: r.probs}); });\n"
        "testData.passwords.forEach(p => { out.passwords.push(aiPasswordPredictability(AI_PWD_MODEL, p)); });\n"
        "testData.messages.forEach(t => { const r = aiPredictEnsemble(AI_MSG_MODEL, AI_MSG_LOGREG_MODEL, t); out.ensemble_messages.push({topClass: r.topClass, probs: r.probs, agree: r.agree}); });\n"
        "testData.osint.forEach(t => { const r = aiPredictEnsemble(AI_OSINT_MODEL, AI_OSINT_LOGREG_MODEL, t); out.ensemble_osint.push({topClass: r.topClass, probs: r.probs, agree: r.agree}); });\n"
        "console.log(JSON.stringify(out));\n"
    )


def main():
    msg_model = load_json("message_classifier.json")
    msg_logreg = load_json("message_classifier_logreg.json")
    osint_model = load_json("osint_risk_classifier.json")
    osint_logreg = load_json("osint_risk_classifier_logreg.json")
    pwd_model = load_json("password_ngram.json")

    html_path = os.path.join(BASE, "Cyber_Granit.html")
    script_code = extract_script(html_path)
    payload = {"messages": TEST_MESSAGES, "osint": TEST_OSINT, "passwords": TEST_PASSWORDS}
    js_code = build_js_harness(script_code, payload)
    stdout = run_js(js_code)
    js_result = json.loads(stdout.strip().splitlines()[-1])

    errors = []

    for i, text in enumerate(TEST_MESSAGES):
        py = eng.predict_nb(msg_model, text)
        js = js_result["messages"][i]
        if py["top_class"] != js["topClass"]:
            errors.append(f"message[{i}] top_class mismatch: py={py['top_class']} js={js['topClass']}")
        for c in msg_model["classes"]:
            if abs(py["probs"][c] - js["probs"][c]) > 1e-6:
                errors.append(f"message[{i}] prob[{c}] mismatch: py={py['probs'][c]:.6f} js={js['probs'][c]:.6f}")

    for i, text in enumerate(TEST_OSINT):
        py = eng.predict_nb(osint_model, text)
        js = js_result["osint"][i]
        if py["top_class"] != js["topClass"]:
            errors.append(f"osint[{i}] top_class mismatch: py={py['top_class']} js={js['topClass']}")

    for i, pwd in enumerate(TEST_PASSWORDS):
        py = eng.score_password_predictability(pwd_model, pwd)
        js = js_result["passwords"][i]
        if abs(py - js) > 0.05:
            errors.append(f"password[{i}] mismatch: py={py} js={js}")

    for i, text in enumerate(TEST_MESSAGES):
        py = eng.predict_ensemble(msg_model, msg_logreg, text)
        js = js_result["ensemble_messages"][i]
        if py["top_class"] != js["topClass"]:
            errors.append(f"ensemble_message[{i}] top_class mismatch: py={py['top_class']} js={js['topClass']}")
        if py["agree"] != js["agree"]:
            errors.append(f"ensemble_message[{i}] agree flag mismatch: py={py['agree']} js={js['agree']}")
        for c in msg_model["classes"]:
            if abs(py["probs"][c] - js["probs"][c]) > 1e-6:
                errors.append(f"ensemble_message[{i}] prob[{c}] mismatch: py={py['probs'][c]:.6f} js={js['probs'][c]:.6f}")

    for i, text in enumerate(TEST_OSINT):
        py = eng.predict_ensemble(osint_model, osint_logreg, text)
        js = js_result["ensemble_osint"][i]
        if py["top_class"] != js["topClass"]:
            errors.append(f"ensemble_osint[{i}] top_class mismatch: py={py['top_class']} js={js['topClass']}")

    if errors:
        print("PARITY TEST FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    print(
        f"Parity OK: {len(TEST_MESSAGES)} messages, {len(TEST_OSINT)} osint-descriptions, "
        f"{len(TEST_PASSWORDS)} passwords, plus NB+LogReg ensemble on both classifiers — "
        f"Python and embedded JavaScript agree."
    )


if __name__ == "__main__":
    main()
