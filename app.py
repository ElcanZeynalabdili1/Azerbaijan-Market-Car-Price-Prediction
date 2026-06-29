import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

CURRENT_YEAR = datetime.now().year

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Avtomobil Qiymət Proqnozu",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Avtomobil Qiymət Proqnozu")
st.caption("turbo.az 2022 məlumatları əsasında hazırlanmış ML modeli")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("car_price_model1.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ `car_price_model.pkl` tapılmadı. Əvvəlcə notebooku çalışdırıb modeli yadda saxla.")
    st.stop()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Avtomobil məlumatları")

marka = st.sidebar.text_input("Marka", value="BMW")
model_name = st.sidebar.text_input("Model", value="5 Series")

year = st.sidebar.number_input(
    "Buraxılış ili", min_value=1990, max_value=CURRENT_YEAR, value=2018, step=1
)

ban_novu = st.sidebar.selectbox("Ban növü", [
    "Sedan", "Hetçbek", "Offroader / SUV", "Universal",
    "Liftbek", "Minivan", "Kupe", "Kabriolet", "Pikap", "Van"
])

suret_qutusu = st.sidebar.selectbox("Sürət qutusu", [
    "Avtomat", "Mexaniki", "Variator", "Robotlaşdırılmış"
])

yanacaq_novu = st.sidebar.selectbox("Yanacaq növü", [
    "Benzin", "Dizel", "Hibrid", "Plug-in Hibrid", "Elektro", "Qaz"
])

reng = st.sidebar.selectbox("Rəng", [
    "Qara", "Ağ", "Gümüşü", "Boz", "Qırmızı", "Mavi",
    "Yaşıl", "Sarı", "Narıncı", "Bənövşəyi", "Qəhvəyi", "Digər"
])

city = st.sidebar.selectbox("Şəhər", [
    "Bakı", "Gəncə", "Sumqayıt", "Mingəçevir", "Naxçıvan",
    "Lənkəran", "Şirvan", "Şəki", "Yevlax", "Digər"
])

muherrik = st.sidebar.number_input("Mühərrik həcmi (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
muherrik_gucu = st.sidebar.number_input("Mühərrik gücü (a.g.)", min_value=50, max_value=1000, value=190, step=5)
yurus = st.sidebar.number_input("Yürüş (km)", min_value=0, max_value=1_000_000, value=120_000, step=1000)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.sidebar.button("💰 Qiyməti hesabla", use_container_width=True, type="primary"):

    input_df = pd.DataFrame([{
        "marka"        : marka,
        "model"        : model_name,
        "car_age"      : CURRENT_YEAR - year,
        "ban_novu"     : ban_novu,
        "reng"         : reng,
        "muherrik"     : float(muherrik),
        "muherrik_gucu": float(muherrik_gucu),
        "suret_qutusu" : suret_qutusu,
        "yanacaq_novu" : yanacaq_novu,
        "log_yurus"    : np.log1p(yurus),
        "city"         : city,
    }])

    log_pred  = model.predict(input_df)[0]
    price_azn = np.expm1(log_pred)
    price_usd = price_azn / 1.7
    price_eur = price_azn / 1.85

    st.success("✅ Proqnoz hazırdır!")

    col1, col2, col3 = st.columns(3)
    col1.metric("🇦🇿 AZN", f"{price_azn:,.0f}")
    col2.metric("🇺🇸 USD", f"{price_usd:,.0f}")
    col3.metric("🇪🇺 EUR", f"{price_eur:,.0f}")

    st.divider()

    # Car summary
    st.subheader("📋 Avtomobil xülasəsi")
    summary = {
        "Marka / Model"   : f"{marka} {model_name}",
        "Buraxılış ili"   : year,
        "Yaş"             : f"{CURRENT_YEAR - year} il",
        "Ban növü"        : ban_novu,
        "Yanacaq"         : yanacaq_novu,
        "Sürət qutusu"    : suret_qutusu,
        "Mühərrik"        : f"{muherrik} L / {muherrik_gucu} a.g.",
        "Yürüş"           : f"{yurus:,} km",
        "Rəng"            : reng,
        "Şəhər"           : city,
    }
    st.table(pd.DataFrame(summary.items(), columns=["Xüsusiyyət", "Dəyər"]))

else:
    st.info("👈 Sol tərəfdən avtomobil məlumatlarını daxil et və **Qiyməti hesabla** düyməsini bas.")

    st.divider()
    st.subheader("ℹ️ Layihə haqqında")
    st.markdown("""
    Bu tətbiq **turbo.az** saytından toplanmış ~31,000 avtomobil elanı əsasında
    maşın öyrənməsi modeli ilə qiymət proqnozu verir.

    **İstifadə edilən modellər:**
    - HistGradientBoostingRegressor
    - RandomForestRegressor
    - GradientBoostingRegressor
    - və digərləri

    **Tech stack:** Python · Scikit-learn · Pandas · Streamlit
    """)
