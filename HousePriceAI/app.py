import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ================= CONFIGURATION =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")

# Feature สำคัญจาก Kaggle ที่ใช้เทรนโมเดล
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "TotRmsAbvGrd",
    "LotArea",
    "1stFlrSF",
    "BedroomAbvGr",
    "KitchenAbvGr"
]

MODEL_PARAMS = {
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
}

USD_TO_THB = 37.0

# ================= STREAMLIT PAGE =================
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 ระบบทำนายราคาบ้านด้วย AI (Decision Tree)")
st.write("💡 ข้อมูลจาก Kaggle House Prices + ราคาทำนายในสกุลเงินบาท")

# ================= MODEL TRAINING =================
def train_model():
    st.info("🚀 กำลังเทรนโมเดลจาก Kaggle โปรดรอสักครู่...")
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

    joblib.dump(model, MODEL_PATH)
    st.success(f"✅ เทรนโมเดลสำเร็จ! (MSE={mse:.2f}, R²={r2:.4f})")
    return model, mse, r2

# ================= LOAD OR TRAIN MODEL =================
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("✅ โหลดโมเดลสำเร็จ")
else:
    model, mse, r2 = train_model()

# ================= USER INPUT =================
st.header("📋 ป้อนข้อมูลบ้านเพื่อทำนายราคา")
col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider("คุณภาพโดยรวมของบ้าน (OverallQual)", 1, 10, 5)
    GarageCars = st.slider("จำนวนรถที่จอดได้", 0, 5, 2)
    FullBath = st.slider("จำนวนห้องน้ำ", 0, 5, 2)
    TotRmsAbvGrd = st.slider("จำนวนห้องทั้งหมดเหนือพื้นดิน", 2, 15, 6)
    BedroomAbvGr = st.slider("จำนวนห้องนอน", 0, 10, 3)

with col2:
    GrLivArea = st.number_input("พื้นที่ใช้สอย (ตร.ฟุต)", 500, 5000, 1500)
    TotalBsmtSF = st.number_input("พื้นที่ชั้นใต้ดิน (ตร.ฟุต)", 0, 3000, 800)
    LotArea = st.number_input("ขนาดที่ดิน (ตร.ฟุต)", 1000, 20000, 8000)
    FirstFlrSF = st.number_input("พื้นที่ชั้น 1 (ตร.ฟุต)", 500, 2000, 1200)
    KitchenAbvGr = st.slider("จำนวนห้องครัว", 0, 5, 1)
    YearBuilt = st.number_input("ปีที่สร้างบ้าน", 1900, 2025, 2005)

input_data = pd.DataFrame([[ 
    OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, 
    YearBuilt, TotRmsAbvGrd, LotArea, FirstFlrSF, BedroomAbvGr, KitchenAbvGr
]], columns=FEATURES)

# ================= PREDICTION =================
if st.button("🔍 ทำนายราคาบ้าน"):
    try:
        log_pred = model.predict(input_data)[0]
        price_usd = np.expm1(log_pred)
        price_thb = price_usd * USD_TO_THB
        st.subheader("💰 ราคาบ้านที่ทำนาย")
        st.write(f"ราคาประเมิน: **฿{price_thb:,.2f} บาท**")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
