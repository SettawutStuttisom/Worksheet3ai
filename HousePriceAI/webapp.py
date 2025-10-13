import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction AI", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction AI")
st.write("ระบบ AI สำหรับพยากรณ์ราคาบ้าน โดยใช้ข้อมูลจริงจาก Kaggle")

# โหลดโมเดลที่เทรนไว้
model = joblib.load("best_model.pkl")

# ส่วนรับค่าจากผู้ใช้
st.subheader("📋 ป้อนข้อมูลบ้านเพื่อทำนายราคา")
overallqual = st.slider("คุณภาพรวมของบ้าน (OverallQual)", 1, 10, 5)
grlivarea = st.number_input("พื้นที่ใช้สอย (GrLivArea)", 500, 5000, 1500)
garagecars = st.slider("จำนวนรถที่จอดได้ (GarageCars)", 0, 4, 2)
totalbsmt = st.number_input("พื้นที่ชั้นใต้ดิน (TotalBsmtSF)", 0, 3000, 800)
fullbath = st.slider("จำนวนห้องน้ำเต็ม (FullBath)", 0, 3, 2)
yearbuilt = st.number_input("ปีที่สร้างบ้าน (YearBuilt)", 1870, 2025, 2000)

if st.button("🔍 ทำนายราคาบ้าน"):
    X_input = pd.DataFrame([[overallqual, grlivarea, garagecars, totalbsmt, fullbath, yearbuilt]],
                           columns=["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"])
    pred = model.predict(X_input)[0]
    st.success(f"ราคาบ้านที่คาดการณ์: **${pred:,.2f}**")
