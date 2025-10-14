import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 📁 ตั้งค่า Path ให้ชัดเจน
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 ระบบทำนายราคาบ้านด้วย AI ")
st.write("โมเดล Random Forest พร้อมระบบโหลด/เทรนอัตโนมัติใน Streamlit")

# แสดงตำแหน่งไฟล์ที่มองหา (สำหรับตรวจสอบ)
st.write("📁 ตำแหน่งที่มองหาไฟล์ train.csv:")
st.code(DATA_PATH)
st.write("📁 ตำแหน่งที่มองหาโมเดล:")
st.code(MODEL_PATH)

# -----------------------------
# ฟังก์ชันเทรนโมเดล
# -----------------------------
def train_model():
    st.info("🚀 กำลังเทรนโมเดลใหม่ โปรดรอสักครู่...")

    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

    if not os.path.exists(DATA_PATH):
        st.error("❌ ไม่พบไฟล์ train.csv ที่ตำแหน่งที่ระบุ")
        st.stop()

    data = pd.read_csv(DATA_PATH)

    features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
    X = data[features]
    y = np.log1p(data["SalePrice"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

    joblib.dump(model, MODEL_PATH)
    st.success(f"✅ เทรนโมเดลสำเร็จ! (MSE={mse:.2f}, R²={r2:.4f})")

    return model

# -----------------------------
# โหลดหรือเทรนโมเดล
# -----------------------------
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

# -----------------------------
# 📋 รับข้อมูลผู้ใช้
# -----------------------------
st.header("📋 ป้อนข้อมูลบ้าน")

col1, col2 = st.columns(2)
with col1:
    OverallQual = st.slider("คุณภาพโดยรวมของบ้าน (OverallQual)", 1, 10, 5)
    GrLivArea = st.number_input("พื้นที่ใช้สอย (ตร.ฟุต)", 500, 5000, 1500)
    GarageCars = st.slider("จำนวนที่จอดรถในโรงรถ", 0, 4, 2)
with col2:
    TotalBsmtSF = st.number_input("พื้นที่ชั้นใต้ดิน (ตร.ฟุต)", 0, 3000, 800)
    FullBath = st.slider("จำนวนห้องน้ำเต็ม (FullBath)", 0, 4, 2)
    YearBuilt = st.number_input("ปีที่สร้างบ้าน", 1900, datetime.datetime.now().year, 2005)

input_data = pd.DataFrame(
    [[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]],
    columns=["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
)

# -----------------------------
# 🔍 ทำนายราคา
# -----------------------------
if st.button("🔍 ทำนายราคาบ้าน"):
    try:
        log_pred = model.predict(input_data)[0]
        prediction = np.expm1(log_pred)

        current_year = datetime.datetime.now().year
        age = current_year - YearBuilt

        # ปรับราคาเพิ่มเติม
        garage_factor = {0: 0.85, 1: 0.93, 2: 1.00, 3: 1.08, 4: 1.15}
        bath_factor = {0: 0.9, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.10}
        prediction *= garage_factor.get(GarageCars, 1.0)
        prediction *= bath_factor.get(FullBath, 1.0)
        if age > 50:
            prediction *= 0.8
        elif age > 30:
            prediction *= 0.9
        elif age < 10:
            prediction *= 1.1

        st.subheader("💰 ผลการทำนายราคาบ้าน")
        st.write(f"ราคาบ้านที่คาดการณ์: **${prediction:,.2f}**")
        st.info(f"🏗️ อายุบ้าน: {age} ปี | 🚗 โรงรถ: {GarageCars} ช่อง | 🛁 ห้องน้ำ: {FullBath} ห้อง")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

st.markdown("---")
st.caption("🤖 พัฒนาโดย เสฎฐวุฒิ | วิชา AI | มข.")
