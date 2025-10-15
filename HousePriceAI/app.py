import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# 📁 CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")

# 📌 Feature ที่ใช้ในการเทรน —> เพิ่มฟีเจอร์ใหม่ได้ที่นี่
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]

# 📌 Parameter เริ่มต้นของโมเดล —> ปรับพารามิเตอร์ได้ที่นี่
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}

# =============================================================================
# 🏡 STREAMLIT PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 ระบบทำนายราคาบ้านด้วย AI")
st.write("โมเดล Random Forest พร้อมระบบโหลด/เทรนอัตโนมัติ")


# =============================================================================
# 🧠 MODEL TRAINING FUNCTION
# =============================================================================
def train_model():
    """เทรนโมเดลใหม่จาก train.csv"""
    st.info("🚀 กำลังเทรนโมเดลใหม่ โปรดรอสักครู่...")

    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

    if not os.path.exists(DATA_PATH):
        st.error("❌ ไม่พบไฟล์ train.csv ที่ตำแหน่งที่ระบุ")
        st.stop()

    # 📊 โหลดข้อมูล
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = np.log1p(data["SalePrice"])

    # ✂️ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🧠 เทรนโมเดล
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # 📈 ประเมินผล
    y_pred = model.predict(X_test)
    mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

    # 💾 บันทึกโมเดล
    joblib.dump(model, MODEL_PATH)

    st.success(f"✅ เทรนโมเดลสำเร็จ! (MSE={mse:.2f}, R²={r2:.4f})")
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
        model = joblib.load(MODEL_PATH)
        st.success("✅ โหลดโมเดลสำเร็จ")
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()


# =============================================================================
# 📝 USER INPUT FORM
# =============================================================================
st.header("📋 ป้อนข้อมูลบ้าน")

col_slider, col_number = st.columns(2)

# 🎚️ Slider ด้านซ้าย (Discrete)
with col_slider:
    OverallQual = st.slider("คุณภาพโดยรวมของบ้าน (OverallQual)", 1, 10, 5)
    GarageCars = st.slider("จำนวนที่จอดรถในโรงรถ", 0, 4, 2)
    FullBath = st.slider("จำนวนห้องน้ำ", 0, 4, 2)

# 🔢 Number input ด้านขวา (Numeric)
with col_number:
    GrLivArea = st.number_input("พื้นที่ใช้สอย (ตร.ฟุต)", 500, 5000, 1500)
    TotalBsmtSF = st.number_input("พื้นที่ชั้นใต้ดิน (ตร.ฟุต)", 0, 3000, 800)
    YearBuilt = st.number_input(
        "ปีที่สร้างบ้าน", 1900, datetime.datetime.now().year, 2005
    )

# 🧾 รวมข้อมูลผู้ใช้เป็น DataFrame
input_data = pd.DataFrame(
    [[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]],
    columns=FEATURES,
)


# =============================================================================
# 🔮 PREDICTION
# =============================================================================
def adjust_price(base_price, garage_cars, full_bath, year_built):
    """ปรับราคาเพิ่มเติมจากตัวแปรต่างๆ"""
    current_year = datetime.datetime.now().year
    age = current_year - year_built

    garage_factor = {0: 0.85, 1: 0.93, 2: 1.00, 3: 1.08, 4: 1.15}
    bath_factor = {0: 0.90, 1: 0.95, 2: 1.00, 3: 1.05, 4: 1.10}

    price = base_price
    price *= garage_factor.get(garage_cars, 1.0)
    price *= bath_factor.get(full_bath, 1.0)

    if age > 50:
        price *= 0.8
    elif age > 30:
        price *= 0.9
    elif age < 10:
        price *= 1.1

    return price, age


if st.button("🔍 ทำนายราคาบ้าน"):
    try:
        log_pred = model.predict(input_data)[0]
        base_price = np.expm1(log_pred)

        final_price, house_age = adjust_price(
            base_price,
            garage_cars=GarageCars,
            full_bath=FullBath,
            year_built=YearBuilt,
        )

        st.subheader("💰 ผลการทำนายราคาบ้าน")
        st.write(f"ราคาบ้านที่คาดการณ์: **${final_price:,.2f}**")
        st.info(
            f"🏗️ อายุบ้าน: {house_age} ปี | 🚗 โรงรถ: {GarageCars} ช่อง | 🛁 ห้องน้ำ: {FullBath} ห้อง"
        )

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")


# =============================================================================
# 🧩 NOTE: จุดเชื่อมต่อสำหรับการขยายในอนาคต
# =============================================================================
# - 📌 เพิ่มฟีเจอร์ใหม่: เพิ่มชื่อ feature ใน FEATURES และเพิ่ม input field ด้านบน
# - 🧠 ปรับโมเดล: เปลี่ยน MODEL_PARAMS / เปลี่ยนโมเดลจาก RandomForest เป็น XGBoost, LightGBM ฯลฯ
# - 💾 เพิ่มระบบ Auto retrain เมื่ออัปโหลดไฟล์ใหม่
# - 📊 แสดงกราฟความสำคัญของฟีเจอร์ (Feature Importance)
# =============================================================================

st.markdown("---")
st.caption("🤖 พัฒนาโดย เสฎฐวุฒิ | วิชา AI | มข.")
