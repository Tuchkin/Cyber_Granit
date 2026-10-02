# -*- coding: utf-8 -*-
"""
Проверки безопасности самого портала Cyber_Granit.html.

1. Политика безопасности контента (CSP): страница не может открывать сетевые соединения
   (fetch, XHR, WebSocket, sendBeacon), подгружать внешние картинки, шрифты и фреймы.
2. Внедрение кода через метаданные фото: анализатор EXIF выводит поля «камера», «модель»
   и «дата съемки» только как текст. Тест собирает настоящий JPEG, в EXIF которого вместо
   модели камеры записан HTML-код, и прогоняет его через exifParse и exifRenderResult.
3. Остальные пути вывода пользовательского текста (подсветка слов в анализаторе) экранируются.
4. Целостность ИИ-модуля: веса, встроенные в HTML, совпадают с эталонными файлами ai_models/*.json
   (сравниваются SHA-256 канонической записи JSON).

Запуск: python tests/test_portal_security.py [путь к html]   (требует установленный Node.js)
"""
import base64
import hashlib
import json
import os
import re
import struct
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_ai_parity import extract_script, run_js  # noqa: E402

HTML_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "Cyber_Granit.html")
PAYLOADS = [
    '<img src=x onerror="alert(1)">',
    '"><svg onload="alert(1)">',
    "<script>alert(1)</script>",
]
DANGEROUS = re.compile(r"<\s*(img|svg|script|iframe)\b", re.I)


def jpeg_with_exif(make, model, date_time):
    """Минимальный JPEG (только маркеры SOI, APP1/Exif, EOI) с тремя текстовыми полями IFD0."""
    entries = [(0x010F, make), (0x0110, model), (0x0132, date_time)]
    data_offset = 8 + 2 + len(entries) * 12 + 4
    ifd, blob = struct.pack("<H", len(entries)), b""
    for tag, text in entries:
        raw = text.encode("latin-1") + b"\x00"
        ifd += struct.pack("<HHII", tag, 2, len(raw), data_offset + len(blob))
        blob += raw
    tiff = b"II*\x00" + struct.pack("<I", 8) + ifd + struct.pack("<I", 0) + blob
    app1 = b"Exif\x00\x00" + tiff
    return b"\xff\xd8" + b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1 + b"\xff\xd9"


WEIGHTS = [
    ("AI_MSG_MODEL", "message_classifier.json"),
    ("AI_MSG_LOGREG_MODEL", "message_classifier_logreg.json"),
    ("AI_OSINT_MODEL", "osint_risk_classifier.json"),
    ("AI_OSINT_LOGREG_MODEL", "osint_risk_classifier_logreg.json"),
    ("AI_PWD_MODEL", "password_ngram.json"),
]


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def check_weights(html):
    errors = []
    for const, name in WEIGHTS:
        m = re.search(r"^const " + const + r" = (.*);$", html, re.M)
        if not m:
            errors.append(f"в HTML нет весов {const}")
            continue
        with open(os.path.join(BASE, "ai_models", name), encoding="utf-8") as f:
            ref = json.load(f)
        if sha256_json(json.loads(m.group(1))) != sha256_json(ref):
            errors.append(f"веса {const} в HTML не совпадают с ai_models/{name}")
    return errors


def check_csp(html):
    m = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    if not m:
        return ["нет мета-тега Content-Security-Policy"]
    pol = {d.split()[0]: d.split()[1:] for d in (x.strip() for x in m.group(1).split(";")) if d}
    errors = []
    for directive in ("default-src", "connect-src", "frame-src", "object-src", "form-action", "base-uri"):
        if pol.get(directive) != ["'none'"]:
            errors.append(f"{directive} должен быть 'none', сейчас {pol.get(directive)}")
    for directive in ("img-src", "font-src"):
        bad = [s for s in pol.get(directive, []) if s not in ("file:", "data:")]
        if bad:
            errors.append(f"{directive}: недопустимые источники {bad}")
    if any("unsafe-eval" in " ".join(v) for v in pol.values()):
        errors.append("политика разрешает eval")
    return errors


def main():
    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()
    errors = check_csp(html) + check_weights(html)

    cases = []
    for i, p in enumerate(PAYLOADS):
        fields = ["samsung", "SM-A546E", "2026:06:12 14:23:07"]
        for k in range(3):
            f3 = list(fields)
            f3[k] = p
            cases.append(base64.b64encode(jpeg_with_exif(*f3)).decode())
    harness = (
        "function stub(){return {value:'',innerHTML:'',style:{},"
        "classList:{add(){},remove(){},contains(){return false;}},"
        "textContent:'',scrollTop:0,scrollHeight:0};}\n"
        "global.document = {getElementById:()=>stub(), querySelectorAll:()=>[], querySelector:()=>stub()};\n"
        "global.window = {scrollTo(){}};\n"
        + extract_script(HTML_PATH)
        + "\nconst cases = " + json.dumps(cases) + ";\n"
        "const payloads = " + json.dumps(PAYLOADS) + ";\n"
        "const out = {exif: [], highlight: []};\n"
        "cases.forEach(b64 => { const buf = Buffer.from(b64, 'base64');\n"
        "  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);\n"
        "  const res = exifParse(ab); const el = stub(); exifRenderResult(el, res);\n"
        "  out.exif.push({parsed: [res.make, res.model, res.dateTime], html: el.innerHTML}); });\n"
        "payloads.forEach(p => { const t = 'Срочно перейди по ссылке ' + p + ' и введи пароль';\n"
        "  const r = aiPredictEnsemble(AI_MSG_MODEL, AI_MSG_LOGREG_MODEL, t);\n"
        "  out.highlight.push(aiHighlightWords(t, r.topFeatures.concat(['img', 'script', 'svg']))); });\n"
        "console.log(JSON.stringify(out));\n"
    )
    res = json.loads(run_js(harness).strip().splitlines()[-1])

    for i, r in enumerate(res["exif"]):
        if not any(p in (r["parsed"] or []) for p in PAYLOADS):
            errors.append(f"exif[{i}]: тестовый файл разобран неверно: {r['parsed']}")
        if DANGEROUS.search(r["html"]):
            errors.append(f"exif[{i}]: HTML из метаданных попал на страницу как код: {r['html'][:120]}")
    for i, h in enumerate(res["highlight"]):
        if DANGEROUS.search(h):
            errors.append(f"highlight[{i}]: текст сообщения не экранирован: {h[:120]}")

    if errors:
        print("PORTAL SECURITY TEST FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"Portal security OK: CSP без сетевых соединений; {len(res['exif'])} вредоносных EXIF-файлов "
          f"и {len(res['highlight'])} сообщений выведены только как текст; "
          f"веса {len(WEIGHTS)} моделей совпадают с эталоном (SHA-256)")


if __name__ == "__main__":
    main()
