import streamlit as st
import re
import math
import random
import time
import os
import sys
import json
import streamlit.components.v1 as components

# Определение базовой директории скрипта и папки с картинками
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
AI_MODELS_DIR = os.path.join(BASE_DIR, "ai_models")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- ИИ-МОДУЛЬ: загрузка заранее обученных офлайн-моделей ---
# Модели обучены локально скриптом ai_models/build_models.py (наивный
# байесовский классификатор + символьная n-граммная языковая модель) и
# сохранены в JSON. Никаких обращений к интернету/внешним API здесь нет —
# только чтение локальных файлов рядом со скриптом.
try:
    from ai_models import nb_engine as ai_engine
    from ai_models.datasets import CHAT_KB as AI_CHAT_KB

    @st.cache_resource
    def load_ai_models():
        def _load(name):
            with open(os.path.join(AI_MODELS_DIR, name), encoding="utf-8") as f:
                return json.load(f)

        msg_model = _load("message_classifier.json")
        msg_logreg = _load("message_classifier_logreg.json")
        osint_model = _load("osint_risk_classifier.json")
        osint_logreg = _load("osint_risk_classifier_logreg.json")
        pwd_model = _load("password_ngram.json")
        chat_index = ai_engine.build_tfidf(AI_CHAT_KB)
        return msg_model, msg_logreg, osint_model, osint_logreg, pwd_model, chat_index

    (AI_MSG_MODEL, AI_MSG_LOGREG_MODEL, AI_OSINT_MODEL, AI_OSINT_LOGREG_MODEL,
     AI_PWD_MODEL, AI_CHAT_INDEX) = load_ai_models()
    AI_AVAILABLE = True
except Exception as ai_load_error:  # модели не обучены (build_models.py ещё не запускался) — не роняем портал
    AI_AVAILABLE = False
    AI_LOAD_ERROR = str(ai_load_error)

AI_MSG_LABELS = {
    "phishing": "ФИШИНГ",
    "social_engineering": "ВЕРБОВКА / СОЦИНЖЕНЕРИЯ",
    "ipso_fake": "ИПсО / ФЕЙК",
    "safe": "БЕЗОПАСНО",
}
AI_OSINT_LABELS = {
    "leak_risk": "РИСК УТЕЧКИ",
    "safe": "БЕЗОПАСНО",
}


def _ai_render_agreement(res, labels):
    nb_label = labels.get(res["nb"]["top_class"], res["nb"]["top_class"])
    lr_label = labels.get(res["logreg"]["top_class"], res["logreg"]["top_class"])
    if res["agree"]:
        st.caption(f"🤝 Две независимые модели (наивный байес и логистическая регрессия) согласны: **{nb_label}**.")
    else:
        st.caption(
            f"⚠️ Модели разошлись во мнении: наивный байес → **{nb_label}**, "
            f"логистическая регрессия → **{lr_label}**. Итоговый вердикт — среднее вероятностей обеих."
        )


def ai_render_message_result(text):
    res = ai_engine.predict_ensemble(AI_MSG_MODEL, AI_MSG_LOGREG_MODEL, text)
    top = res["top_class"]
    conf = round(res["probs"][top] * 100, 1)
    if top == "safe":
        st.success(f"✅ ИИ не обнаружил признаков угрозы — вероятнее всего, безопасное сообщение ({conf}%).")
    else:
        st.error(f"🚨 ИИ обнаружил признаки: **{AI_MSG_LABELS[top]}** (уверенность {conf}%).")
    for c in ["phishing", "social_engineering", "ipso_fake", "safe"]:
        st.progress(res["probs"][c], text=f"{AI_MSG_LABELS[c]}: {round(res['probs'][c]*100,1)}%")
    if res["top_features"]:
        st.caption("Слова, повлиявшие на решение ИИ: " + ", ".join(res["top_features"]))
    _ai_render_agreement(res, AI_MSG_LABELS)


def ai_render_osint_result(text):
    res = ai_engine.predict_ensemble(AI_OSINT_MODEL, AI_OSINT_LOGREG_MODEL, text)
    top = res["top_class"]
    if top == "safe":
        st.success("✅ ИИ не выявил признаков оперативной утечки в описании.")
    else:
        risk = round(res["probs"].get("leak_risk", 0) * 100, 1)
        st.error(f"🚨 ИИ считает публикацию рискованной: вероятность утечки {risk}%.")
    for c in ["leak_risk", "safe"]:
        st.progress(res["probs"][c], text=f"{AI_OSINT_LABELS[c]}: {round(res['probs'][c]*100,1)}%")
    if res["top_features"]:
        st.caption("На это обратил внимание ИИ: " + ", ".join(res["top_features"]))
    _ai_render_agreement(res, AI_OSINT_LABELS)

# --- НАСТРОЙКИ СТРАНИЦЫ И СТИЛИ ---
st.set_page_config(page_title="Кибер-Гранит | ВПК", page_icon="🛡️", layout="wide")

# Применяем кастомные стили для стилистики милитари/защиты
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #4CAF50;}
    .stAlert {background-color: #1e1e1e; color: #ffffff;}
    .css-1d391kg {background-color: #262730;} /* Sidebar */
    .stProgress .st-bo {background-color: #4CAF50;}
    .success-text {color: #4CAF50; font-weight: bold;}
    .warning-text {color: #FFC107; font-weight: bold;}
    .error-text {color: #F44336; font-weight: bold;}
    .info-box {
        background-color: #2e3b32;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #3b2e2e;
        border-left: 5px solid #F44336;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .hero-banner {
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# PLACEHOLDER_NAVIGATION_START
st.sidebar.title("🛡️ Кибер-Гранит")
st.sidebar.markdown("**Учебный портал цифровой безопасности для ВПК**")
st.sidebar.divider()

st.sidebar.markdown("""
<div style='color: #4CAF50; font-size: 0.9em; margin-bottom: 15px;'>
    <b>Автор:</b> Голубева Дарина<br>
    <i>(224 СШ г. Минска, ВПК Гранит)</i><br><br>
    <b>Руководитель:</b> Янцевич Михаил<br>
    <i>(Военная академия Республики Беларусь)</i>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

page = st.sidebar.radio("Навигация по разделам:", 
    ["Главная", 
     "Законодательство РБ",
     "ИПсО и Фейки",
     "ОСИНТ и Соцсети", 
     "Пароль-контроль",
     "Анти-Фишинг",
     "ИИ-Ассистент",
     "Кибер-Полигон"])

# Автоматическая прокрутка наверх при смене раздела
if 'current_page' not in st.session_state:
    st.session_state.current_page = page

if st.session_state.current_page != page:
    st.session_state.current_page = page
    st.rerun()

# Принудительный скролл наверх (100% рабочий метод для Streamlit)
components.html(
    "<script>var m = window.parent.document.querySelector('.main'); if (m) { m.scrollTo(0,0); }</script>",
    height=0
)

st.sidebar.divider()
st.sidebar.info("Разработано для повышения уровня информационной защиты курсантов военно-патриотических клубов.")

# --- РАЗДЕЛ: ГЛАВНАЯ ---
def page_home():
    # Попытка загрузить сгенерированную эмблему ВПК Гранит
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Картинка адаптируется под ширину колонки, чтобы не сжиматься
            st.image(os.path.join(IMAGES_DIR, "cyber-granite-emblem.png"), use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")

    # Уменьшили размер заголовков и выровняли по центру
    st.markdown("<h2 style='text-align: center;'>🛡️ Добро пожаловать на проект «Кибер-Гранит»</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #bbbbbb;'>Твоя безопасность в сети — это безопасность твоего подразделения и твоей страны.</h4>", unsafe_allow_html=True)
    st.write("")
    
    st.write("""
    Современные войны выигрываются не только на поле боя, но и в информационном пространстве. 
    Твой смартфон, твои социальные сети, твои пароли — это передовая линия обороны. 
    Здесь ты научишься:
    * Защищать свои данные от перехвата и взлома.
    * Распознавать методы социальной инженерии и вербовки.
    * Понимать, как работает разведка по открытым источникам (OSINT).
    * Соблюдать законы Республики Беларусь в цифровой среде.
    """)
    
    st.info("👈 Выбери раздел в меню слева, чтобы начать подготовку.")
    
    st.markdown("---")
    
    # НОВЫЙ БЛОК: Статистика киберпреступности (Единый день информирования, ноябрь 2025)
    st.markdown("<h3 style='text-align: center; color: #4CAF50;'>📊 Кибер-обстановка в Республике Беларусь (ноябрь 2025)</h3>", unsafe_allow_html=True)
    st.write("""
    По материалам Единого дня информирования, кибератаки стали одной из главных угроз национальной безопасности.
    **Беларусь входит в тройку стран СНГ** по количеству кибератак (2–3 место по разным отчётам 2024–2025 гг.), и злоумышленники активно используют методы социальной инженерии, фишинг и кибербуллинг.
    """)
    
    # Вставляем картинку с графиком секторов
    try:
        col_img1, col_img2, col_img3 = st.columns([1, 3, 1])
        with col_img2:
            st.image(os.path.join(IMAGES_DIR, "stats-sectors-2025.png"), caption="Распределение кибератак по отраслям в Республике Беларусь", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.markdown("""
        <div class="danger-box" style="text-align: center;">
            <h2 style="color: #F44336; margin: 0;">22%</h2>
            <b>Атак на Госсектор</b><br>
            <span style="font-size: 0.8em;">(Промышленность — 14%, Финансы — 11%)</span>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown("""
        <div class="danger-box" style="text-align: center;">
            <h2 style="color: #F44336; margin: 0;">57%</h2>
            <b>Случаев утечки данных</b><br>
            <span style="font-size: 0.8em;">В основном — личная информация граждан</span>
        </div>
        """, unsafe_allow_html=True)
    with col_stat3:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h2 style="color: #4CAF50; margin: 0;">№70</h2>
            <b>Нац. индекс кибербезопасности NCSI, 2024</b><br>
            <span style="font-size: 0.8em;">Создан Национальный центр кибербезопасности</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Учебных модулей", value="7")
    with col2:
        st.metric(label="Уровень угрозы в сети", value="ВЫСОКИЙ")
    with col3:
        st.metric(label="Статус подготовки", value="Ожидает")

# --- РАЗДЕЛ: ЗАКОНОДАТЕЛЬСТВО РБ ---
def page_laws():
    st.title("⚖️ Законодательство РБ: Информационная безопасность")
    st.markdown("### Незнание закона не освобождает от ответственности")
    
    # Заглавная иллюстрация раздела Законодательство РБ
    try:
        st.image(os.path.join(IMAGES_DIR, "legislation-belarus-cyber.png"), caption="Законодательство Республики Беларусь в киберсфере", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
        
    st.write("""
    Интернет-пространство в Республике Беларусь регулируется законами так же строго, как и реальная жизнь. 
    Курсант ВПК обязан знать правовые нормы и не допускать правонарушений, даже "по глупости" или "из любопытства".
    """)
    
    st.header("Уголовный Кодекс (УК РБ)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.error("**Статья 349. Несанкционированный доступ к компьютерной информации**")
        st.write("Взлом чужой страницы ВКонтакте, чтение чужих переписок, использование чужого пароля без разрешения. **Наказание: вплоть до лишения свободы.**")
        
        st.error("**Статья 342. Организация и подготовка действий, грубо нарушающих общественный порядок**")
        st.write("Участие в закрытых деструктивных чатах, где координируются незаконные массовые мероприятия, перекрытие дорог. **Наказание: до 4 лет лишения свободы.**")
        
    with col2:
        st.error("**Статья 203-1. Незаконные действия в отношении информации о частной жизни и персональных данных**")
        st.write("Слив (деанон) личных данных, адресов, телефонов командиров, сотрудников милиции, должностных лиц или товарищей. **Наказание: вплоть до лишения свободы.**")
        
        st.error("**Статья 361-1. Создание экстремистского формирования либо участие в нем**")
        st.write("Администрирование или активное участие в деятельности групп, каналов, признанных экстремистскими формированиями. **Строгое уголовное наказание.**")

    st.markdown("---")
    st.header("Кодекс об Административных Правонарушениях (КоАП РБ)")
    st.markdown("""
    <div class="danger-box">
    <b>Статья 19.11 КоАП РБ (Распространение экстремистских материалов)</b><br>
    Самая частая ошибка подростков. Наказание грозит за:
    <ul>
        <li>Подписку на Telegram-каналы или Instagram-аккаунты, внесенные в Республиканский список экстремистских материалов.</li>
        <li>Репост или пересылку другу сообщения из такого канала (даже ради смеха).</li>
        <li>Сохранение картинок или видео с логотипами деструктивных каналов в сохраненках или на стене ВК.</li>
    </ul>
    <i>Последствия: штраф, конфискация телефона/компьютера или административный арест.</i>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("🛡️ **Совет:** Регулярно проверяйте свои подписки, удаляйте старые репосты и сомнительные сохраненные картинки. Список экстремистских материалов открыто публикуется на сайте Министерства информации РБ.")


# --- РАЗДЕЛ: ОСИНТ И СОЦСЕТИ ---
def page_osint():
    st.title("👁️‍🗨️ Соцсети и ОСИНТ: Невидимый фронт")
    st.markdown("### Как твои фото могут выдать позиции и подставить товарищей")
    
    # Заглавная иллюстрация раздела ОСИНТ
    try:
        st.image(os.path.join(IMAGES_DIR, "osint-social-media-analysis.png"), caption="Анализ цифрового следа военнослужащего: угрозы открытых данных (OSINT)", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
    
    st.markdown("""
    <div class="danger-box">
    <b>OSINT (Разведка по открытым источникам)</b> — это сбор данных о противнике через анализ публичной информации: соцсетей, фотографий, форумов и баз данных. 
    Для злоумышленника или иностранной разведки твоя страница ВКонтакте или Instagram — это кладезь ценной информации.
    </div>
    """, unsafe_allow_html=True)
    
    st.header("📸 1. Правило заднего фона")
    st.write("""
    **Курсант в форме всегда привлекает внимание.** Делая селфи или снимая видео для TikTok, ты можешь случайно засветить:
    * **Расположение части или клуба** (ориентиры, здания, номера кабинетов).
    * **Военную технику** (ее тип, количество, состояние, бортовые номера).
    * **Лица командиров и товарищей** (которые могут быть не согласны на публикацию).
    * **Документы или расписания** (лежащие на столе на заднем плане).
    
    *Приказ:* Перед тем как сделать фото в форме, осмотрись на 360 градусов. Убедись, что в кадр не попадет ничего, кроме тебя и нейтральной стены.
    """)
    
    st.header("📍 2. Невидимый предатель: Геометки (EXIF)")
    st.write("""
    Каждый раз, когда ты делаешь фото на смартфон, в сам файл (в его свойства) зашиваются так называемые EXIF-данные. 
    Они включают точные **GPS-координаты места съемки, время и модель твоего телефона**.
    Даже если ты стер название города в посте, метаданные фото могут выдать твое точное местоположение с точностью до метра.
    """)
    
    st.info("🛠️ **Как защититься:** Зайди в настройки камеры своего телефона и **ОТКЛЮЧИ сохранение геопозиции (тегов местоположения)** для фото и видео.")

    st.markdown("---")
    st.header("🔬 Проверь своё фото на геометки прямо сейчас")
    st.write("Это не ИИ, а честный разбор EXIF-метаданных файла — тот же принцип, которым пользуются настоящие OSINT-аналитики. Файл обрабатывается локально на этом же компьютере и никуда не отправляется в интернет.")
    exif_file = st.file_uploader("Выбери JPEG-файл:", type=["jpg", "jpeg"], key="exif_uploader")
    if exif_file is not None:
        try:
            from PIL import Image
            img = Image.open(exif_file)
            exif = img.getexif()
            if not exif:
                st.success("✅ В файле не найдено EXIF-метаданных (либо они уже были удалены — это правильная практика перед публикацией).")
            else:
                gps_ifd = exif.get_ifd(0x8825)
                make = exif.get(0x010F)
                model = exif.get(0x0110)
                date_time = exif.get(0x0132)

                if gps_ifd and 2 in gps_ifd and 4 in gps_ifd:
                    def _dms_to_decimal(dms, ref):
                        deg, minu, sec = dms
                        dec = float(deg) + float(minu) / 60 + float(sec) / 3600
                        return -dec if ref in ("S", "W") else dec

                    lat = _dms_to_decimal(gps_ifd[2], gps_ifd.get(1, "N"))
                    lon = _dms_to_decimal(gps_ifd[4], gps_ifd.get(3, "E"))
                    st.error(f"🚨 В файле зашиты GPS-координаты места съёмки: **{lat:.6f}, {lon:.6f}**. "
                             f"Именно так злоумышленник может определить точное расположение объекта на фото, "
                             f"даже если геометка не видна на самом изображении.")
                    st.caption(f"Ссылка для проверки (нужен интернет): https://www.google.com/maps?q={lat:.6f},{lon:.6f}")
                else:
                    st.success("✅ GPS-координаты в файле не найдены.")

                meta = []
                if make or model:
                    meta.append(f"камера/устройство: {(make or '')} {(model or '')}".strip())
                if date_time:
                    meta.append(f"дата съёмки: {date_time}")
                if meta:
                    st.info("ℹ️ Дополнительно в метаданных: " + "; ".join(meta) + ".")
        except Exception as e:
            st.warning(f"Не удалось разобрать файл: {e}")

    st.header("🔒 3. Базовая настройка Telegram")
    st.write("Твой Telegram должен быть крепостью. Зайди в Настройки -> Конфиденциальность и установи:")
    st.write("✅ **Номер телефона:** Кто видит - Никто. Кто может найти по номеру - Мои контакты.")
    st.write("✅ **Звонки (Peer-to-Peer):** Только контакты (иначе при звонке можно вычислить твой IP-адрес).")
    st.write("✅ **Фотография профиля:** Мои контакты (чтобы чужие не могли использовать твое фото для поиска по лицу (Face Search)).")

    st.markdown("---")
    st.header("🧠 ИИ-скоринг риска публикации")
    try:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(os.path.join(IMAGES_DIR, "osint-ai-photo-scanner.png"), caption="ИИ-сканер выявляет риск утечки по описанию фото или поста", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
    st.write("Опишите словами свою будущую публикацию (что на фото, что написано на заднем плане, есть ли геометка) — ИИ оценит риск оперативной утечки.")
    if AI_AVAILABLE:
        osint_ai_text = st.text_area("Описание публикации:", key="ai_osint_input", placeholder="Например: селфи на фоне доски с расписанием дежурств и картой района...")
        if st.button("🔍 Оценить риск через ИИ", key="ai_osint_btn") and osint_ai_text.strip():
            ai_render_osint_result(osint_ai_text)
    else:
        st.info("ИИ-модели не обучены — см. раздел «ИИ-Ассистент».")

# --- РАЗДЕЛ: ПАРОЛЬ-КОНТРОЛЬ ---
def page_passwords():
    st.title("🔐 Пароль-контроль: Твой цифровой бронежилет")
    st.markdown("### Тренажер оценки стойкости пароля")
    
    # Заглавная иллюстрация раздела Пароль-контроль
    try:
        st.image(os.path.join(IMAGES_DIR, "password-control-vault.png"), caption="Цифровой сейф: Надежный пароль — основа кибербезопасности", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
    
    st.markdown("""
    <div class="info-box">
    Пароль — это первый рубеж обороны твоих данных. 
    Слабый пароль взламывается программами-переборщиками (брутфорсом) по словарям за миллисекунды. 
    Если твой аккаунт вскроют, злоумышленники смогут писать от твоего имени командирам, рассылать фейки товарищам или вымогать деньги у родственников.
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ ВНИМАНИЕ: Это учебный полигон. **НЕ ВВОДИТЕ** здесь свои реальные пароли, которые вы используете! Придумайте тестовый пример для проверки.")

    if AI_AVAILABLE:
        try:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image(os.path.join(IMAGES_DIR, "ai-password-strength-scan.png"), caption="Ниже к обычной проверке добавлена AI-оценка предсказуемости", width="stretch")
        except Exception as e:
            st.error(f"Ошибка загрузки картинки: {e}")

    # --- ЛОГИКА ОЦЕНКИ ПАРОЛЯ ---
    def evaluate_password(pwd):
        score = 0
        feedback = []
        charset = 0
        time_to_crack = "Мгновенно"
        
        # Длина
        if len(pwd) < 8:
            feedback.append("❌ Длина менее 8 символов. Это критически мало!")
        else:
            score += 1
            if len(pwd) >= 12: score += 1
            if len(pwd) >= 16: score += 1
            
        # Сложность
        if re.search(r"[a-zа-я]", pwd): 
            score += 1
            charset += 32
        else: 
            feedback.append("❌ Нет строчных букв.")
            
        if re.search(r"[A-ZА-Я]", pwd): 
            score += 1
            charset += 32
        else: 
            feedback.append("❌ Нет заглавных букв.")
            
        if re.search(r"[0-9]", pwd): 
            score += 1
            charset += 10
        else: 
            feedback.append("❌ Нет цифр.")
            
        if re.search(r"[!@#$%^&*()_\-=\[\]{};':\"\\|,.<>/?]", pwd): 
            score += 1
            charset += 30
        else: 
            feedback.append("❌ Нет спецсимволов (!@#$%^&*).")
            
        # Паттерны и словари (имитация)
        bad_patterns = r"(123|qwe|йцу|password|пароль|qwerty|admin|2000|2023|2024)"
        if re.search(bad_patterns, pwd.lower()):
            score -= 2
            feedback.append("❌ Обнаружен частый шаблон (123, qwe, год). Это легко взломать по словарю!")
            
        # Расчет энтропии (сложности перебора)
        if len(pwd) > 0 and charset > 0:
            entropy = len(pwd) * math.log2(charset)
            combinations = charset ** len(pwd)
            seconds = combinations / 100_000_000_000
            
            if seconds < 1: time_to_crack = "Доли секунды 🔴"
            elif seconds < 60: time_to_crack = f"{int(seconds)} секунд 🔴"
            elif seconds < 3600: time_to_crack = f"{int(seconds/60)} минут 🟠"
            elif seconds < 86400: time_to_crack = f"{int(seconds/3600)} часов 🟠"
            elif seconds < 31536000: time_to_crack = f"{int(seconds/86400)} дней 🟡"
            elif seconds < 3153600000: time_to_crack = f"{int(seconds/31536000)} лет 🟢"
            else: time_to_crack = "Тысячелетия 🛡️"
        
        final_score = max(0, min(100, int((score / 6) * 100)))
        return final_score, feedback, time_to_crack

    test_password = st.text_input("Введите тренировочный пароль:", type="default", placeholder="Например: MyP@ssw0rd!2024")
    
    if test_password:
        score, feedback, time_to_crack = evaluate_password(test_password)
        
        st.write("### Анализ стойкости:")
        st.progress(score / 100.0)
        
        if score < 40:
            st.error(f"Уровень защиты: СЛАБЫЙ ({score}%)")
        elif score < 80:
            st.warning(f"Уровень защиты: СРЕДНИЙ ({score}%)")
        else:
            st.success(f"Уровень защиты: ОТЛИЧНЫЙ ({score}%)")
            
        st.metric(label="Примерное время на взлом (Брутфорс атака)", value=time_to_crack)

        if AI_AVAILABLE:
            ai_pct = ai_engine.score_password_predictability(AI_PWD_MODEL, test_password)
            st.metric(label="🧠 AI-оценка предсказуемости", value=f"{ai_pct}%")
            st.progress(ai_pct / 100.0)
            if ai_pct >= 65:
                st.warning("⚠️ ИИ-модель (обучена на типовых слабых паролях) считает этот пароль похожим на распространённые шаблоны — даже если формально он проходит проверку по длине и составу символов.")
            elif ai_pct >= 30:
                st.info("ИИ-модель находит отдельные узнаваемые фрагменты, но в целом структура пароля близка к нетиповой.")
            else:
                st.success("✅ ИИ-модель не находит в пароле узнаваемых шаблонов — структура символов статистически близка к случайной.")

        if feedback:
            st.write("**Уязвимости вашего пароля:**")
            for f in feedback:
                st.write(f)
        elif score >= 80:
            st.write("<span class='success-text'>✅ Отличная работа! Этот пароль выдержит любую атаку перебором.</span>", unsafe_allow_html=True)
            
    st.divider()
    
    st.header("📚 Теория: Как правильно хранить ключи от цифровой жизни")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💡 Парольная фраза")
        st.write("""
        Вместо сложных бессмысленных наборов лучше использовать **Парольные фразы**. 
        Это несколько случайных слов, разделенных спецсимволом или цифрой.
        Они длинные (огромная энтропия), но легко запоминаются.
        
        *Пример:* `Танк_Поле_Красный_Орел!99`
        """)
        
    with col2:
        st.subheader("🗄️ Менеджеры паролей")
        st.write("""
        У тебя не должен быть один пароль от всего! Храни пароли в **Менеджере паролей** (KeePassXC, Bitwarden).
        Это защищенный сейф. Тебе нужно запомнить всего ОДИН мастер-пароль, чтобы открыть сейф, а остальные ключи программа будет подставлять сама.
        """)
        
    st.markdown("""
    <div class="info-box">
    <h3>🔑 Двухфакторная аутентификация (2FA) - Правило двух ключей</h3>
    Даже самый сложный пароль могут украсть вирусом. ВЕЗДЕ должна быть включена 2FA. <br>
    Это значит, что для входа нужен не только пароль (то, что ты <b>знаешь</b>), но и код из СМС/приложения (то, чем ты <b>владеешь</b>). Без телефона хакер не зайдет.
    </div>
    """, unsafe_allow_html=True)

# --- РАЗДЕЛ: ФИШИНГ ---
def page_phishing():
    st.title("🎣 Анти-Фишинг: Распознавание вербовки и обмана")
    st.markdown("### Защита от социальной инженерии")
    
    # Заглавная иллюстрация раздела Анти-Фишинг
    try:
        st.image(os.path.join(IMAGES_DIR, "anti-phishing-shield.png"), caption="Анти-фишинг: Блокировка попыток выудить конфиденциальные данные", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
        
    st.write("Фишинг — это метод обмана, при котором диверсант выдает себя за доверенное лицо, чтобы выведать данные, заразить устройство или заставить совершить преступление.")

    # Вставляем картинку с демографией жертв мошенников
    try:
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(os.path.join(IMAGES_DIR, "fraud-demographics.png"), caption="Гендерно-возрастное распределение жертв кибермошенников", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")

    st.subheader("⚠️ Главные угрозы для курсанта ВПК")
    
    # Вставляем картинку с угрозами для детей
    try:
        st.image(os.path.join(IMAGES_DIR, "kids-cyber-threats.png"), caption="Основные схемы угроз для детей и подростков (фишинг, груминг, кибербуллинг)", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
        
    col1, col2 = st.columns(2)
    with col1:
        st.error("**«Фейковый командир»**\n\nМошенники создают аккаунт с фото вашего руководителя. Пишут с незнакомого номера: *«Срочно проголосуй за меня по ссылке»* или *«Сбрось списки отряда»*.")
        st.warning("**Вербовка под прикрытием**\n\nНеизвестные предлагают «легкий заработок» (сделать фото объектов, перевести деньги). Это прямой путь к уголовной ответственности за пособничество экстремизму.")
    with col2:
        st.info("**Взлом через любопытство**\n\nВам присылают файл `Компромат_на_клуб.exe`. Скачивание гарантированно устанавливает на телефон шпионскую программу.")
        st.success("**Кликбейт и паника**\n\n*«Ваш аккаунт взломан!»*. Цель — отключить ваше критическое мышление и заставить действовать на эмоциях.")

    st.markdown("---")
    st.subheader("🕵️‍♂️ Интерактивный тренажер: Вычисли диверсанта")

    # Scenario 1
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**Сообщение в Telegram от 'Командир Иванов':**")
    st.info("Боец, срочно! Наш клуб участвует в голосовании. Перейди по ссылке http://vpk-vote-belarus.xyz/login и введи свой пароль от Telegram для подтверждения голоса.")
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("✅ Безопасно, выполняю", key="phish_1_safe"):
        st.error("💥 ОШИБКА! Ваш аккаунт угнан. Командиры никогда не просят вводить пароли на сторонних сайтах.")
    if col_btn2.button("🚨 Фишинг! Заблокировать", key="phish_1_bad"):
        st.success("🎖️ ОТЛИЧНО! Вы распознали классический фишинг. Правильно: связаться с командиром по сотовой связи.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Scenario 2
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**Сообщение от неизвестного:**")
    st.info("Привет! Есть подработка на 100 рублей. Нужно просто сфотографировать трансформаторную будку возле вашей станции. Оплата на крипту. Берешься?")
    
    col_btn3, col_btn4 = st.columns(2)
    if col_btn3.button("✅ Легкие деньги", key="phish_2_safe"):
        st.error("💥 КРИТИЧЕСКАЯ ОШИБКА! Это попытка вербовки для диверсии. Немедленно сообщите в милицию!")
    if col_btn4.button("🚨 Вербовка! Доложить", key="phish_2_bad"):
        st.success("🎖️ ИДЕАЛЬНО! Вы проявили бдительность. Вражеские спецслужбы используют подростков втемную для сбора разведданных.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🧠 ИИ-анализ произвольного сообщения")
    st.write("Вставьте своё сообщение — тот же ИИ-модуль, что и в разделе «ИИ-Ассистент», определит тип угрозы.")
    if AI_AVAILABLE:
        phish_ai_text = st.text_area("Текст сообщения:", key="ai_phish_input", placeholder="Вставьте текст подозрительного сообщения...")
        if st.button("🔍 Проверить через ИИ", key="ai_phish_btn") and phish_ai_text.strip():
            ai_render_message_result(phish_ai_text)
    else:
        st.info("ИИ-модели не обучены — см. раздел «ИИ-Ассистент».")

# --- РАЗДЕЛ: КИБЕР-ПОЛИГОН (КВЕСТ) ---
def page_polygon():
    st.title("🎯 Кибер-Полигон: Операция «Цифровой Щит»")
    
    # Заглавная иллюстрация раздела Кибер-Полигон
    try:
        st.image(os.path.join(IMAGES_DIR, "cyber-polygon-quest.png"), caption="Учебный киберполигон «Бастион-1»", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
    
    col_title1, col_title2 = st.columns([3, 1])
    with col_title1:
        st.markdown("### Интерактивный командно-штабной квест")
    with col_title2:
        if st.button("🔄 Сбросить прогресс", use_container_width=True):
            st.session_state.quest_stage = 0
            st.session_state.quest_mistakes = 0
            st.session_state.quest_feedback = None
            st.session_state.quest_stage_completed = False
            st.rerun()

    # Инициализация переменных состояния квеста
    if 'quest_stage' not in st.session_state:
        st.session_state.quest_stage = 0
    if 'quest_mistakes' not in st.session_state:
        st.session_state.quest_mistakes = 0
    if 'quest_feedback' not in st.session_state:
        st.session_state.quest_feedback = None
    if 'quest_feedback_type' not in st.session_state:
        st.session_state.quest_feedback_type = None
    if 'quest_stage_completed' not in st.session_state:
        st.session_state.quest_stage_completed = False

    def set_feedback(msg, msg_type, completed):
        st.session_state.quest_feedback = msg
        st.session_state.quest_feedback_type = msg_type
        st.session_state.quest_stage_completed = completed

    def next_stage(stage_num):
        st.session_state.quest_stage = stage_num
        st.session_state.quest_feedback = None
        st.session_state.quest_feedback_type = None
        st.session_state.quest_stage_completed = False

    # Убрали вывод сообщения отсюда, чтобы показывать его внутри этапа
    
    # Stage 0: Briefing
    if st.session_state.quest_stage == 0:
        st.markdown("""
        <div class="info-box">
        <b>ВВОДНАЯ:</b> Вы заступили на дежурство в штабе ВПК «Гранит». 
        Поступили разведданные о готовящейся комплексной кибератаке на инфраструктуру клуба.
        Ваша задача: принимать оперативные решения, отразить атаки и вычислить угрозу.
        Каждое решение влияет на итоговую оценку.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎖️ ПРИСТУПИТЬ К ВЫПОЛНЕНИЮ БОЕВОЙ ЗАДАЧИ", use_container_width=True):
            st.session_state.quest_stage = 1
            st.session_state.quest_mistakes = 0
            st.rerun()

    # Stage 1: Phishing
    elif st.session_state.quest_stage == 1:
        st.subheader("Этап 1: Входящее сообщение")
        try:
            st.image(os.path.join(IMAGES_DIR, "phishing-email-screen.png"), caption="Входящее сообщение на терминале дежурного: тема письма вызывает подозрения.", width="stretch")
        except Exception as e:
            st.error(f"Ошибка загрузки картинки: {e}")
        st.write("На дежурный компьютер штаба приходит письмо с пометкой «СРОЧНО»:")
        st.info("**От:** admin@m1n-oborony-rb.com\n\n**Тема:** Сверка списков личного состава ВПК на 2025 год.\n\n**Вложение:** Списки_Гранит_2025.scr\n\nБоец, немедленно открой вложение, проверь свои данные и доложи об исполнении!")

        if AI_AVAILABLE and st.button("🤖 Запросить ИИ-анализ сообщения", key="ai_hint_1"):
            ai_render_message_result(
                "От: admin@m1n-oborony-rb.com. Тема: Сверка списков личного состава ВПК на 2025 год. "
                "Вложение: Списки_Гранит_2025.scr. Боец, немедленно открой вложение, проверь свои данные "
                "и доложи об исполнении!"
            )

        if not st.session_state.quest_stage_completed:
            st.write("**Ваш приказ?**")
            if st.button("📂 Скачать и открыть файл, приказ есть приказ", key="q1_1"):
                st.session_state.quest_mistakes += 1
                set_feedback("💥 КРИТИЧЕСКАЯ ОШИБКА! Расширение .scr — это исполняемый файл вируса. Вы заразили штабной компьютер трояном. Враг получил доступ к сети.", "error", False)
                st.rerun()
            if st.button("✉️ Ответить на письмо и попросить уточнить детали", key="q1_2"):
                st.session_state.quest_mistakes += 1
                set_feedback("⚠️ ОШИБКА! Ответив, вы подтвердили злоумышленникам, что ваш почтовый ящик активен. Ждите шквал спама. А адрес отправителя - подделка (обратите внимание на m1n-oborony вместо min).", "warning", False)
                st.rerun()
            if st.button("🚨 Не открывать, удалить письмо и доложить командиру о попытке фишинга", key="q1_3"):
                set_feedback("✅ ВЕРНОЕ РЕШЕНИЕ! Вы заметили поддельный адрес (m1n-oborony) и опасное вложение (.scr). Атака отбита.", "success", True)
                st.rerun()
        
        # Вывод результата внутри этапа
        if st.session_state.quest_feedback:
            st.markdown("---")
            if st.session_state.quest_feedback_type == 'error': st.error(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'warning': st.warning(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'success': st.success(st.session_state.quest_feedback)

        if st.session_state.quest_stage_completed:
            if st.button("Продолжить операцию ➡️", key="next_1"):
                next_stage(2)
                st.rerun()

    # Stage 2: OSINT
    elif st.session_state.quest_stage == 2:
        st.subheader("Этап 2: Утечка данных (OSINT)")
        try:
            st.image(os.path.join(IMAGES_DIR, "cadet-selfie-geotag-opsec.png"), caption="Пример опасного фото: нарушение правила заднего фона и раскрытие геопозиции.", width="stretch")
        except Exception as e:
            st.error(f"Ошибка загрузки картинки: {e}")
        st.write("Аналитики сообщили, что враг пытается вычислить координаты нашего запасного пункта сбора через соцсети курсантов. Вы проверяете страницу курсанта Иванова.")
        st.write("Какая из его последних публикаций является критической уязвимостью?")
        
        if not st.session_state.quest_stage_completed:
            if st.button("📸 Фото в камуфляже на фоне городского памятника с геометкой центра города", key="q2_1"):
                st.session_state.quest_mistakes += 1
                set_feedback("⚠️ Нежелательно, но памятник - публичное место. Это не выдает секретные координаты штаба. Ищите более серьезную угрозу.", "warning", False)
                st.rerun()
            if st.button("📸 Селфи в штабе, где на заднем плане видна доска с расписанием дежурств и картой района", key="q2_2"):
                set_feedback("✅ ВЕРНО! Правило заднего фона нарушено. Враг может увеличить фото, прочитать расписание и изучить карту. Публикация немедленно удалена.", "success", True)
                st.rerun()
            if st.button("📸 Текстовый пост: \"Устал после марш-броска, завтра снова на тренировку\"", key="q2_3"):
                st.session_state.quest_mistakes += 1
                set_feedback("💥 ОШИБКА. В этом тексте нет конкретики. Вы упускаете реальную утечку на фото!", "error", False)
                st.rerun()
                
        # Вывод результата внутри этапа
        if st.session_state.quest_feedback:
            st.markdown("---")
            if st.session_state.quest_feedback_type == 'error': st.error(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'warning': st.warning(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'success': st.success(st.session_state.quest_feedback)

        if st.session_state.quest_stage_completed:
            if st.button("Продолжить операцию ➡️", key="next_2"):
                next_stage(3)
                st.rerun()

    # Stage 3: Passwords
    elif st.session_state.quest_stage == 3:
        st.subheader("Этап 3: Брутфорс-атака")
        st.write("Враг начал атаку перебором паролей на сервер базы данных ВПК. Система требует немедленно сменить мастер-пароль администратора на максимально надежный.")
        st.write("**Какой пароль вы установите?**")
        
        if not st.session_state.quest_stage_completed:
            if st.button("🔑 VPK_Granit_2025!", key="q3_1"):
                st.session_state.quest_mistakes += 1
                set_feedback("💥 ОШИБКА. Слишком предсказуемо. Хакерские словари первым делом проверяют названия организаций и текущий год.", "error", False)
                st.rerun()
            if st.button("🔑 Zxcvbnm123456!@#", key="q3_2"):
                st.session_state.quest_mistakes += 1
                set_feedback("💥 ОШИБКА. Это шаблонный пароль (ряды на клавиатуре). Программы для брутфорса взломают его за секунды.", "error", False)
                st.rerun()
            if st.button("🔑 Bronya_Granit_Rubezh_Sever_99!", key="q3_3"):
                set_feedback("✅ ВЕРНО! Использована длинная парольная фраза. Энтропия этого пароля не позволит взломать его даже суперкомпьютеру в ближайшие сотни лет. Атака захлебнулась.", "success", True)
                st.rerun()
                
        # Вывод результата внутри этапа
        if st.session_state.quest_feedback:
            st.markdown("---")
            if st.session_state.quest_feedback_type == 'error': st.error(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'warning': st.warning(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'success': st.success(st.session_state.quest_feedback)

        if st.session_state.quest_stage_completed:
            if st.button("Продолжить операцию ➡️", key="next_3"):
                next_stage(4)
                st.rerun()

    # Stage 4: IPSO
    elif st.session_state.quest_stage == 4:
        st.subheader("Этап 4: Информационно-психологическая атака (ИПсО)")
        try:
            st.image(os.path.join(IMAGES_DIR, "fake-news-ipso-smartphone.png"), caption="Пример информационно-психологической атаки (ИПсО) через популярный мессенджер.", width="stretch")
        except Exception as e:
            st.error(f"Ошибка загрузки картинки: {e}")
        st.write("В крупном региональном Telegram-канале выходит пост: *«СРОЧНО! ВПК Гранит расформировывают, а всех курсантов заставляют подписать документы о неразглашении! Мой сын в панике! МАКСИМАЛЬНЫЙ РЕПОСТ!!!»*")
        st.write("Родители курсантов начинают обрывать телефоны штаба. Ваши действия?")

        if AI_AVAILABLE and st.button("🤖 Запросить ИИ-анализ сообщения", key="ai_hint_4"):
            ai_render_message_result(
                "СРОЧНО! ВПК Гранит расформировывают, а всех курсантов заставляют подписать документы "
                "о неразглашении! Мой сын в панике! МАКСИМАЛЬНЫЙ РЕПОСТ!!!"
            )

        if not st.session_state.quest_stage_completed:
            if st.button("💬 Написать с личного аккаунта в комментарии под постом: «Вы врете! Я курсант, у нас все хорошо!»", key="q4_1"):
                st.session_state.quest_mistakes += 1
                set_feedback("⚠️ ОШИБКА. Вступая в спор с ботами, вы поднимаете активность поста (алгоритмы Telegram покажут его большему числу людей). Троллям только этого и надо.", "warning", False)
                st.rerun()
            if st.button("📢 Скопировать этот пост и переслать в чат клуба с вопросом: «Ребята, это правда???»", key="q4_2"):
                st.session_state.quest_mistakes += 1
                set_feedback("💥 КРИТИЧЕСКАЯ ОШИБКА! Вы сами стали инструментом ИПсО и помогли врагу распространить панику внутри подразделения.", "error", False)
                st.rerun()
            if st.button("🛡️ Сделать скриншот, отправить командиру для публикации официального опровержения и кинуть жалобу на пост", key="q4_3"):
                set_feedback("✅ ИДЕАЛЬНО! Вы не поддались эмоциям, остановили распространение фейка и передали информацию по команде для грамотного контрудара.", "success", True)
                st.rerun()
                
        # Вывод результата внутри этапа
        if st.session_state.quest_feedback:
            st.markdown("---")
            if st.session_state.quest_feedback_type == 'error': st.error(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'warning': st.warning(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'success': st.success(st.session_state.quest_feedback)

        if st.session_state.quest_stage_completed:
            if st.button("Продолжить операцию ➡️", key="next_4"):
                next_stage(5)
                st.rerun()

    # Stage 5: Physical Security
    elif st.session_state.quest_stage == 5:
        st.subheader("Этап 5: Опасная находка")
        try:
            st.image(os.path.join(IMAGES_DIR, "suspicious-flash-drive-checkpoint.png"), caption="Неизвестный накопитель, обнаруженный на территории части. Потенциальная угроза BadUSB.", width="stretch")
        except Exception as e:
            st.error(f"Ошибка загрузки картинки: {e}")
        st.write("Во время патрулирования территории клуба возле КПП вы находите оставленную кем-то USB-флешку. На ней маркером написано «ДМБ 2024. Фотографии».")
        st.write("Ваши действия?")
        
        if not st.session_state.quest_stage_completed:
            if st.button("💻 Вставить в штабной компьютер, чтобы посмотреть фото и найти владельца", key="q5_1"):
                st.session_state.quest_mistakes += 1
                set_feedback("💥 КРИТИЧЕСКАЯ ОШИБКА! Это классическая атака «BadUSB» (Троянский конь). При подключении флешки вредоносная программа мгновенно установилась на ПК и дала врагу доступ к сети штаба.", "error", False)
                st.rerun()
            if st.button("📱 Подключить через переходник к своему личному смартфону (его не так жалко)", key="q5_2"):
                st.session_state.quest_mistakes += 1
                set_feedback("⚠️ ОШИБКА! Ваш смартфон — это тоже компьютер. Вирус похитит ваши личные данные, пароли и контакты товарищей по клубу.", "warning", False)
                st.rerun()
            if st.button("🛡️ Не подключать никуда, передать дежурному офицеру для уничтожения или проверки на изолированном стенде", key="q5_3"):
                set_feedback("✅ ОТЛИЧНО! Вы пресекли попытку физического проникновения во внутреннюю сеть ВПК.", "success", True)
                st.rerun()
                
        if st.session_state.quest_feedback:
            st.markdown("---")
            if st.session_state.quest_feedback_type == 'error': st.error(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'warning': st.warning(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'success': st.success(st.session_state.quest_feedback)

        if st.session_state.quest_stage_completed:
            if st.button("Продолжить операцию ➡️", key="next_5"):
                next_stage(6)
                st.rerun()

    # Stage 6: Vishing
    elif st.session_state.quest_stage == 6:
        st.subheader("Этап 6: Звонок «из Министерства» (Вишинг)")
        try:
            st.image(os.path.join(IMAGES_DIR, "vishing-call-ministry.png"), caption="Входящий вызов с подменой номера (вишинг) с целью получения конфиденциальной информации.", width="stretch")
        except Exception as e:
            st.error(f"Ошибка загрузки картинки: {e}")
        st.write("На ваш личный телефон поступает звонок. Голос звучит строго: *«Курсант, это проверка связи из Министерства обороны. На ваш номер сейчас придет секретный код подтверждения готовности. Немедленно продиктуйте его!»*")
        st.write("Вам действительно приходит SMS с кодом. Ваши действия?")

        if AI_AVAILABLE and st.button("🤖 Запросить ИИ-анализ сообщения", key="ai_hint_6"):
            ai_render_message_result(
                "Курсант, это проверка связи из Министерства обороны. На ваш номер сейчас придет секретный "
                "код подтверждения готовности. Немедленно продиктуйте его!"
            )

        if not st.session_state.quest_stage_completed:
            if st.button("📱 Продиктовать код, так как звонят из высшего руководства", key="q6_1"):
                st.session_state.quest_mistakes += 1
                set_feedback("💥 КРИТИЧЕСКАЯ ОШИБКА! Это был мошенник. Код из SMS — это пароль от вашего аккаунта Telegram или интернет-банкинга. Вы сами отдали врагу ключи.", "error", False)
                st.rerun()
            if st.button("🤔 Сказать, что код не пришел, и попытаться потянуть время", key="q6_2"):
                st.session_state.quest_mistakes += 1
                set_feedback("⚠️ ОШИБКА. Вступая в разговор с социальным инженером, вы даете ему шанс применить психологическое давление. Не играйте с ними.", "warning", False)
                st.rerun()
            if st.button("📵 Молча положить трубку и заблокировать номер", key="q6_3"):
                set_feedback("✅ ВЕРНО! Настоящие командиры и ведомства НИКОГДА не запрашивают коды из SMS по телефону. Вы успешно отбили атаку.", "success", True)
                st.rerun()
                
        if st.session_state.quest_feedback:
            st.markdown("---")
            if st.session_state.quest_feedback_type == 'error': st.error(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'warning': st.warning(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'success': st.success(st.session_state.quest_feedback)

        if st.session_state.quest_stage_completed:
            if st.button("Продолжить операцию ➡️", key="next_6"):
                next_stage(7)
                st.rerun()

    # Stage 7: AI Voice Deepfake
    elif st.session_state.quest_stage == 7:
        st.subheader("Этап 7: ИИ-подделка голоса (дипфейк)")
        st.write("Тебе звонит человек с незнакомого номера. Голос — один в один командир: та же интонация, та же манера речи.")
        st.info("*«Боец, это я, разговор голосом. Срочно, без вопросов — продиктуй мне пароль от общего чата отряда, нужно разослать важный приказ, а я не могу зайти со своего телефона!»*")

        if AI_AVAILABLE and st.button("🤖 Запросить ИИ-анализ сообщения", key="ai_hint_7"):
            ai_render_message_result(
                "Боец, это я, срочно, без вопросов, продиктуй мне пароль от общего чата отряда, нужно "
                "разослать важный приказ, а я не могу зайти со своего телефона."
            )

        if not st.session_state.quest_stage_completed:
            st.write("**Голос звучит абсолютно достоверно. Твои действия?**")
            if st.button("🎙️ Сразу продиктовать: голос точно командира, я его узнаю", key="q7_1"):
                st.session_state.quest_mistakes += 1
                set_feedback("💥 КРИТИЧЕСКАЯ ОШИБКА! Современные нейросети клонируют голос человека по 10–15 секундам записи из открытого видео или сторис. Голос — больше не доказательство личности, даже когда «звучит один в один».", "error", False)
                st.rerun()
            if st.button("🤔 Остаться на линии и попытаться проверить его вопросами о делах отряда", key="q7_2"):
                st.session_state.quest_mistakes += 1
                set_feedback("⚠️ РИСКОВАННО. Подготовленный злоумышленник (или ИИ в реальном времени) может знать общедоступные детали и потянуть время, пока вы теряете бдительность. Разговор по этому каналу ненадёжен в принципе.", "warning", False)
                st.rerun()
            if st.button("📵 Вежливо сказать, что перезвонишь, положить трубку и связаться с командиром по уже сохранённому номеру", key="q7_3"):
                set_feedback("✅ ВЕРНО! Единственная надёжная проверка при подозрении на голосовой дипфейк — независимый канал связи: перезвонить по заранее известному номеру, написать в проверенный чат или уточнить лично. Атака отбита.", "success", True)
                st.rerun()

        if st.session_state.quest_feedback:
            st.markdown("---")
            if st.session_state.quest_feedback_type == 'error': st.error(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'warning': st.warning(st.session_state.quest_feedback)
            elif st.session_state.quest_feedback_type == 'success': st.success(st.session_state.quest_feedback)

        if st.session_state.quest_stage_completed:
            if st.button("Завершить операцию и получить оценку 🏁", key="next_7"):
                next_stage(8)
                st.rerun()

    # Stage 8: Results
    elif st.session_state.quest_stage == 8:
        st.header("🏁 ОПЕРАЦИЯ ЗАВЕРШЕНА")
        mistakes = st.session_state.quest_mistakes
        
        st.write(f"**Допущено тактических ошибок:** {mistakes}")
        st.markdown("---")
        
        st.subheader("🎖️ Итог аттестации:")
        if mistakes == 0:
            st.markdown("<h2 style='color:#FFD700;'>🌟 Майор Кибербезопасности</h2>", unsafe_allow_html=True)
            st.success("Блестящая работа! Вы действовали хладнокровно и профессионально. Инфраструктура ВПК надежно защищена. Командование объявляет вам благодарность.")
        elif mistakes <= 2:
            st.markdown("<h2 style='color:#C0C0C0;'>🛡️ Сержант Информационных Войск</h2>", unsafe_allow_html=True)
            st.warning("Операция выполнена, но были допущены ошибки. В реальном бою это могло стоить утечки данных. Требуется дополнительная подготовка, но потенциал отличный.")
        else:
            st.markdown("<h2 style='color:#cd7f32;'>🪖 Рядовой-Новобранец (Отправлен на переподготовку)</h2>", unsafe_allow_html=True)
            st.error("Штаб взломан, данные утекли в сеть, личный состав деморализован. Вы провалили задание. Тщательно изучите теорию во всех разделах портала и попробуйте снова!")
            
        if st.button("🔄 Пройти операцию заново"):
            st.session_state.quest_stage = 0
            st.session_state.quest_mistakes = 0
            st.session_state.quest_feedback = None
            st.session_state.quest_stage_completed = False
            st.rerun()

# --- РАЗДЕЛ: ИПсО И ФЕЙКИ ---
def page_ipso():
    st.title("🧠 ИПсО: Информационно-психологическое противоборство")
    st.markdown("### Битва за умы: Как распознать фейки и манипуляции")
    
    # Заглавная иллюстрация раздела ИПсО
    try:
        st.image(os.path.join(IMAGES_DIR, "ipso-information-warfare-shield.png"), caption="Информационная безопасность — щит от психологических манипуляций", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
    
    st.markdown("""
    <div class="danger-box">
    <b>Информационно-психологические специальные операции (ИПсО)</b> — это целенаправленные действия противника в информационном пространстве. 
    Их цель: посеять панику, подорвать доверие к государству и армии, спровоцировать незаконные действия и расколоть общество.
    </div>
    """, unsafe_allow_html=True)
    
    st.header("🦠 Анатомия фейка: Как нас обманывают")
    col1, col2 = st.columns(2)
    with col1:
        st.error("**1. Эмоциональная накачка**\\n\\nТекст вызывает сильный страх, гнев или жалость. Большое количество восклицательных знаков, капслока: «СРОЧНО!», «МАКСИМАЛЬНЫЙ РЕПОСТ!», «ОТ НАС ВСЕ СКРЫВАЮТ!».")
        st.error("**2. Отсутствие источника**\\n\\nИнформация подается от лица «анонимного источника», «знакомого из министерства» или брата/свата. Нет ссылки на официальные ведомства (МВД, Минобороны, БелТА).")
    with col2:
        st.error("**3. Фейковые фото и видео (Дипфейки)**\\n\\nИспользование старых кадров из других стран, выдаваемых за текущие события в Беларуси. Монтаж аудио или генерация лиц с помощью нейросетей.")
        st.error("**4. Призыв к немедленному действию**\\n\\nВас торопят: «Срочно снимайте деньги!», «Бегите из города!», «Выходите на улицу!». Спешка отключает критическое мышление.")

    st.markdown("---")
    st.header("🛡️ Правила информационной гигиены (Фактчекинг)")
    st.success("✅ **Доверяй только официальным источникам.** Информацию о происшествиях проверяй в каналах профильных министерств (МВД, МЧС, Минобороны) и государственных СМИ.")
    st.success("✅ **Используй обратный поиск.** Сомневаешься в фото? Загрузи его в Яндекс.Картинки или Google Images (поиск по картинке). Часто оказывается, что «вчерашний пожар в Минске» — это фото 5-летней давности из другой страны.")
    st.success("✅ **Правило 24 часов.** Не делай репост шокирующей новости сразу. Подожди сутки. За это время фейк либо опровергнут, либо подтвердят официально.")

    st.markdown("---")
    st.subheader("🕵️‍♂️ Тренажер Фактчекера")
    st.write("Какая из этих новостей в Telegram-канале является фейком (ИПсО)?")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("""
        <div style='background-color: #2e3b32; padding: 15px; border-radius: 5px; border: 1px solid #4CAF50;'>
        <b>Источник:</b> Официальный канал Минобороны РБ<br><br>
        <i>«С 15 по 20 числа на полигоне Гожский пройдут плановые тактические учения с боевой стрельбой. Просим граждан сохранять спокойствие и не приближаться к району учений.»</i>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Это фейк!", key="fake_1"):
            st.error("❌ Ошибка. Это официальное сообщение ведомства с четким указанием сроков, места и без эмоциональной окраски.")
            
    with col_f2:
        st.markdown("""
        <div style='background-color: #3b2e2e; padding: 15px; border-radius: 5px; border: 1px solid #F44336;'>
        <b>Источник:</b> Анонимный канал "Правда Тут"<br><br>
        <i>«СРОЧНО!!! 🔥🔥🔥 Только что поступила инфа от надежного источника в штабе! Завтра всех курсантов ВПК экстренно отправляют на границу! МАКСИМАЛЬНЫЙ РЕПОСТ, пока не удалили!!!»</i>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Это фейк!", key="fake_2"):
            st.success("🎖️ ВЕРНО! Классическое ИПсО: капслок, эмодзи, «надежный источник», нагнетание паники и призыв к репосту.")

    st.markdown("---")
    st.subheader("🧠 ИИ-детектор фейков")
    st.write("Вставьте текст любого поста или сообщения — ИИ оценит, похож ли он на ИПсО/фейк, фишинг, вербовку или безопасное сообщение.")
    if AI_AVAILABLE:
        ipso_ai_text = st.text_area("Текст новости или поста:", key="ai_ipso_input", placeholder="Вставьте текст новости или поста...")
        if st.button("🔍 Проверить через ИИ", key="ai_ipso_btn") and ipso_ai_text.strip():
            ai_render_message_result(ipso_ai_text)
    else:
        st.info("ИИ-модели не обучены — см. раздел «ИИ-Ассистент».")

# --- РАЗДЕЛ: ИИ-АССИСТЕНТ ---
def page_ai():
    st.title("🤖 ИИ-Ассистент «Кибер-Гранит.ИИ»")
    st.markdown("### Офлайн-модули искусственного интеллекта: работают локально, без интернета")

    try:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(os.path.join(IMAGES_DIR, "ai-hub-neural-shield.png"), caption="Модуль искусственного интеллекта «Кибер-Гранит.ИИ» — офлайн-анализ угроз", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")

    if not AI_AVAILABLE:
        st.error(
            "ИИ-модели ещё не обучены. Запустите `python ai_models/build_models.py`, "
            f"затем перезапустите приложение.\n\nТехническая причина: {AI_LOAD_ERROR}"
        )
        return

    st.markdown("""
    <div class="info-box">
    На портале работают пять собственных ИИ-моделей, обученных заранее и полностью автономных — <b>без единого обращения к интернету или внешним серверам</b>: классификатор угроз в сообщениях — ансамбль из 2 независимых алгоритмов (наивный байес + логистическая регрессия, точность на отложенной тестовой выборке 98%), скоринг риска OSINT-публикаций — тоже ансамбль из 2 моделей (100%), и символьная n-граммная языковая модель для оценки предсказуемости паролей. Методика обучения, датасеты и метрики — в файле <code>ai_models/MODEL_CARD.md</code>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("🧠 ИИ-анализатор сообщений")
    try:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(os.path.join(IMAGES_DIR, "ai-message-classifier-terminal.png"), caption="Классификатор угроз в сообщениях", width="stretch")
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
    st.write("Вставьте любое подозрительное сообщение — ИИ определит тип угрозы (фишинг / вербовка / ИПсО / безопасно) и покажет ключевые слова, повлиявшие на решение.")
    msg_text = st.text_area("Текст сообщения:", key="ai_hub_msg", placeholder="Например: Боец, срочно перейди по ссылке и подтверди пароль...")
    if st.button("🔍 Проанализировать", key="ai_hub_msg_btn") and msg_text.strip():
        ai_render_message_result(msg_text)

    st.markdown("---")
    st.header("👁️ ИИ-скоринг риска OSINT-публикации")
    st.write("Опишите словами фото или пост, который собираетесь опубликовать — ИИ оценит риск утечки оперативной информации.")
    osint_text = st.text_area("Описание публикации:", key="ai_hub_osint", placeholder="Например: селфи в штабе, на фоне видна доска с расписанием дежурств...")
    if st.button("🔍 Оценить риск", key="ai_hub_osint_btn") and osint_text.strip():
        ai_render_osint_result(osint_text)

    st.markdown("---")
    st.header("💬 ИИ-консультант портала")
    st.write("Задайте вопрос по кибербезопасности, законодательству РБ или правилам ОСИНТ — ИИ найдёт наиболее подходящий ответ из базы знаний портала (TF-IDF + косинусное сходство).")

    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    suggestions = [
        "Что делать, если нашёл флешку?",
        "Как включить 2FA?",
        "Какая статья за деанон командира?",
        "Как проверить новость на фейк?",
    ]
    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        if col.button(s, key=f"ai_suggest_{s}"):
            st.session_state.ai_chat_history.append(("user", s))
            st.session_state.ai_chat_history.append(("bot", ai_engine.chat_answer(AI_CHAT_KB, AI_CHAT_INDEX, s)))

    question = st.text_input("Введите вопрос:", key="ai_chat_input")
    if st.button("Спросить", key="ai_chat_btn") and question.strip():
        st.session_state.ai_chat_history.append(("user", question))
        st.session_state.ai_chat_history.append(("bot", ai_engine.chat_answer(AI_CHAT_KB, AI_CHAT_INDEX, question)))

    for role, payload in st.session_state.ai_chat_history[-12:]:
        if role == "user":
            st.markdown(f"**🧑 Вы:** {payload}")
        else:
            if payload["matched"]:
                st.markdown(f"**🤖 ИИ** _(похожий вопрос в базе: «{payload['q']}»)_:  \n{payload['text']}")
            else:
                st.markdown(f"**🤖 ИИ:** {payload['text']}")

    st.markdown("---")
    st.header("⚙️ Как это устроено (прозрачность ИИ)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Классификатор угроз и OSINT-риска — ансамбль из 2 моделей")
        st.write("**Модель A:** наивный байесовский классификатор на символьных триграммах слов. **Модель B:** многоклассовая логистическая регрессия (softmax), обученная градиентным спуском на тех же признаках — принципиально иной, дискриминативный алгоритм. Итоговый вердикт — среднее вероятностей обеих моделей; расхождение мнений честно показывается пользователю. Обучены на 250+ примерах (сообщения) и 100+ примерах (OSINT), реализованы «с нуля» на чистом Python, без сторонних ML-библиотек.")
    with col2:
        st.subheader("AI-модель паролей")
        st.write("Символьная n-граммная языковая модель (марковская цепь 3-го порядка), обученная на корпусе типовых слабых паролей-паттернов. Оценивает статистическое сходство пароля с распространёнными шаблонами.")

    st.warning("⚠️ ИИ-модуль — тренажёр и инструмент поддержки решений, а не замена бдительности и правил, изученных в разделах портала. Все вычисления выполняются локально на компьютере пользователя.")


# --- РОУТИНГ СТРАНИЦ ---
if page == "Главная":
    page_home()
elif page == "Законодательство РБ":
    page_laws()
elif page == "ИПсО и Фейки":
    page_ipso()
elif page == "ОСИНТ и Соцсети":
    page_osint()
elif page == "Пароль-контроль":
    page_passwords()
elif page == "Анти-Фишинг":
    page_phishing()
elif page == "ИИ-Ассистент":
    page_ai()
elif page == "Кибер-Полигон":
    page_polygon()