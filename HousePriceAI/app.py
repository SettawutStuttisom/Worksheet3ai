import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# 📁 CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt"
]

MODEL_PARAMS = {
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
}

USD_TO_THB = 37.0

# =============================================================================
# 🏡 STREAMLIT PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 ระบบทำนายราคาบ้าน Homey")
st.write("💡 ข้อมูลจาก Kaggle House Prices")

# ตัวแปรเก็บค่าความแม่นยำ
model_mse = None
model_r2 = None

# =============================================================================
# 🧠 MODEL TRAINING FUNCTION
# =============================================================================
def train_model():
    global model_mse, model_r2
    st.info("🚀 กำลังเทรนโมเดลใหม่ โปรดรอสักครู่...")
    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

    if not os.path.exists(DATA_PATH):
        st.error("❌ ไม่พบไฟล์ train.csv")
        st.stop()

    data = pd.read_csv(DATA_PATH)

    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        st.error(f"❌ ฟีเจอร์เหล่านี้ไม่มีในไฟล์: {missing}")
        st.stop()

    X = data[FEATURES]
    y = np.log1p(data["SalePrice"])

    model = DecisionTreeRegressor(**MODEL_PARAMS)
    model.fit(X, y)

    y_pred = model.predict(X)
    model_mse = mean_squared_error(np.expm1(y), np.expm1(y_pred))
    model_r2 = r2_score(np.expm1(y), np.expm1(y_pred))

    joblib.dump((model, model_mse, model_r2), MODEL_PATH)
    st.success(f"✅ เทรนโมเดลสำเร็จ! ( R²={model_r2:.4f})")
    return model

# =============================================================================
# 📥 LOAD OR TRAIN MODEL
# =============================================================================
if not os.path.exists(MODEL_PATH):
    if os.path.exists(DATA_PATH):
        model = train_model()
    else:
        st.error("❌ ไม่พบไฟล์ data/train.csv")
        st.stop()
else:
    try:
        model, model_mse, model_r2 = joblib.load(MODEL_PATH)
        st.success("✅ โหลดโมเดลสำเร็จ")
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

# =============================================================================
# 📝 USER INPUT FORM
# =============================================================================
st.header("📋 ป้อนข้อมูลบ้าน")

col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider("คุณภาพโดยรวมของบ้าน (OverallQual)", 1, 10, 5)
    GarageCars = st.slider("จำนวนรถที่จอดได้", 0, 5, 2)
    FullBath = st.slider("จำนวนห้องน้ำ", 0, 5, 2)

with col2:
    GrLivArea = st.number_input("พื้นที่ใช้สอย (ตร.ฟุต)", 500, 5000, 1500)
    TotalBsmtSF = st.number_input("พื้นที่ชั้นใต้ดิน (ตร.ฟุต)", 0, 3000, 800)
    YearBuilt = st.number_input("ปีที่สร้างบ้าน", 1900, 2025, 2005)

# รวมข้อมูลผู้ใช้เป็น DataFrame
input_data = pd.DataFrame(
    [[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]],
    columns=FEATURES
)

# =============================================================================
# 🔘 BUTTON FOR PREDICTION
# =============================================================================
if st.button("🔍 ทำนายราคาบ้าน (บาท)"):
    try:
        log_pred = model.predict(input_data)[0]
        base_price_usd = np.expm1(log_pred)
        final_price_thb = base_price_usd * USD_TO_THB

        st.subheader("💰 ผลการทำนายราคาบ้าน")
        st.write(f"ราคาบ้านที่คาดการณ์: **฿{final_price_thb:,.2f} บาท**")
        st.info(
            f"🏗️ ปีที่สร้าง: {YearBuilt} | 🚗 โรงรถ: {GarageCars} ช่อง | 🛁 ห้องน้ำ: {FullBath} ห้อง | 📐 พื้นที่ใช้สอย: {GrLivArea} ft² | พื้นที่ชั้นใต้ดิน: {TotalBsmtSF} ft²"
        )

        # 📈 แสดงค่าความแม่นยำของโมเดล
        st.subheader("📈 ค่าความแม่นยำของโมเดล")

        st.write(f"**R² Score:** {model_r2:.4f}")

        # 🧩 Feature & Parameter
        st.subheader("🧩 ข้อมูลโมเดล Decision Tree")
        st.write("**Feature ที่ใช้เทรน:**")
        st.write(FEATURES)
        st.write("**Parameter ของโมเดล:**")
        st.write(MODEL_PARAMS)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

st.markdown("---")
st.caption("🤖 พัฒนาโดย เสฎฐวุฒิ | วิชา AI | มข.")
