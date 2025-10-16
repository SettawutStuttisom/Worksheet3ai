import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
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

# =============================================================================
# 🏡 STREAMLIT PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 ระบบทำนายราคาบ้านด้วย AI (Decision Tree)")

# =============================================================================
# 🧠 MODEL TRAINING FUNCTION
# =============================================================================
def train_model():
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
    return model

# =============================================================================
# 📥 LOAD OR TRAIN MODEL
# =============================================================================
if not os.path.exists(MODEL_PATH):
    model = train_model()
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

col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider("คุณภาพโดยรวมของบ้าน (OverallQual)", 1, 10, 5)
    GarageCars = st.slider("จำนวนรถที่จอดได้", 0, 5, 2)
    FullBath = st.slider("จำนวนห้องน้ำ", 0, 5, 2)
    TotRmsAbvGrd = st.slider("จำนวนห้องเหนือพื้นดิน", 2, 15, 6)
    BedroomAbvGr = st.slider("จำนวนห้องนอน", 0, 10, 3)

with col2:
    GrLivArea_m2 = st.number_input("พื้นที่ใช้สอย (ตร.เมตร)", 50, 500, 150)
    TotalBsmtSF_m2 = st.number_input("พื้นที่ชั้นใต้ดิน (ตร.เมตร)", 0, 300, 80)
    LotArea_m2 = st.number_input("ขนาดที่ดิน (ตร.เมตร)", 100, 5000, 800)
    FirstFlr_m2 = st.number_input("พื้นที่ชั้น 1 (ตร.เมตร)", 50, 500, 120)
    KitchenAbvGr = st.slider("จำนวนห้องครัว", 0, 5, 1)
    YearBuilt = st.number_input(
        "ปีที่สร้างบ้าน", 1900, datetime.now().year, 2005
    )

# แปลงจาก m² → ft²
m2_to_ft2 = 10.7639
GrLivArea = GrLivArea_m2 * m2_to_ft2
TotalBsmtSF = TotalBsmtSF_m2 * m2_to_ft2
LotArea = LotArea_m2 * m2_to_ft2
FirstFlrSF = FirstFlr_m2 * m2_to_ft2

input_data = pd.DataFrame([[ 
    OverallQual,
    GrLivArea,
    GarageCars,
    TotalBsmtSF,
    FullBath,
    YearBuilt,
    TotRmsAbvGrd,
    LotArea,
    FirstFlrSF,
    BedroomAbvGr,
    KitchenAbvGr
]], columns=FEATURES)

# =============================================================================
# 🔮 PREDICTION FUNCTION
# =============================================================================
def adjust_price(base_price, garage_cars, full_bath, bedroom, kitchen,
                 GrLivArea_m2, TotalBsmtSF_m2, LotArea_m2, FirstFlr_m2, year_built):
    current_year = datetime.now().year
    age = current_year - year_built

    price = base_price

    # ปรับตามจำนวนรถและห้องน้ำ
    garage_factor = {0:0.85,1:0.93,2:1.00,3:1.08,4:1.15,5:1.18}
    bath_factor = {0:0.90,1:0.95,2:1.00,3:1.05,4:1.10,5:1.12}
    price *= garage_factor.get(garage_cars,1.0)
    price *= bath_factor.get(full_bath,1.0)

    # ห้องนอน
    if bedroom <= 2:
        price *= 0.95
    elif 3 <= bedroom <= 4:
        price *= 1.00
    elif 5 <= bedroom <= 6:
        price *= 1.05
    else:
        price *= 1.08

    # ห้องครัว
    if kitchen == 0:
        price *= 0.85
    elif kitchen == 1:
        price *= 1.00
    elif kitchen == 2:
        price *= 1.05
    else:
        price *= 1.10

    # อายุบ้าน (ปีสร้างใหม่ → ราคาสูงขึ้น)
    price *= 1 + (year_built - 2000)/100

    # ปรับตามพื้นที่
    price *= 1 + (GrLivArea_m2 - 150)/500       # พื้นที่ใช้สอย
    price *= 1 + (TotalBsmtSF_m2 - 80)/300      # ชั้นใต้ดิน
    price *= 1 + (LotArea_m2 - 800)/2000        # ขนาดที่ดิน
    price *= 1 + (FirstFlr_m2 - 120)/200        # ชั้น 1

    return price, age

# =============================================================================
# 🔘 BUTTON FOR PREDICTION
# =============================================================================
if st.button("🔍 ทำนายราคาบ้าน (บาท)"):
    try:
        log_pred = model.predict(input_data)[0]
        base_price_usd = np.expm1(log_pred)
        final_price_usd, house_age = adjust_price(
            base_price_usd,
            garage_cars=GarageCars,
            full_bath=FullBath,
            bedroom=BedroomAbvGr,
            kitchen=KitchenAbvGr,
            GrLivArea_m2=GrLivArea_m2,
            TotalBsmtSF_m2=TotalBsmtSF_m2,
            LotArea_m2=LotArea_m2,
            FirstFlr_m2=FirstFlr_m2,
            year_built=YearBuilt
        )

        final_price_thb = final_price_usd * USD_TO_THB

        st.subheader("💰 ผลการทำนายราคาบ้าน")
        st.write(f"ราคาบ้านที่คาดการณ์: **฿{final_price_thb:,.2f} บาท**")
        st.info(
            f"🏗️ อายุบ้าน: {house_age} ปี | 🚗 โรงรถ: {GarageCars} ช่อง | 🛁 ห้องน้ำ: {FullBath} ห้อง | 🛏️ ห้องนอน: {BedroomAbvGr} ห้อง | 🍳 ห้องครัว: {KitchenAbvGr} ห้อง | 📐 พื้นที่ใช้สอย: {GrLivArea_m2} m²"
        )
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

st.markdown("---")
st.caption("🤖 พัฒนาโดย เสฎฐวุฒิ | วิชา AI | มข.")
