# app.py
import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 ระบบทำนายราคาบ้านด้วย AI (เวอร์ชันสมบูรณ์)")
st.write("โมเดล Random Forest + การปรับราคาตามคุณสมบัติของบ้าน")

# -----------------------------
# โหลดโมเดล
# -----------------------------
try:
    model = joblib.load("model/best_model.pkl")
    st.success("✅ โหลดโมเดลสำเร็จ")
except:
    st.error("❌ ไม่พบไฟล์ model/best_model.pkl กรุณารัน trainmodel.py ก่อน")
    st.stop()

# -----------------------------
# ส่วนรับข้อมูลผู้ใช้
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

# -----------------------------
# เตรียมข้อมูล
# -----------------------------
input_data = pd.DataFrame(
    [[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]],
    columns=["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
)

# -----------------------------
# ปุ่มทำนาย
# -----------------------------
if st.button("🔍 ทำนายราคาบ้าน"):
    try:
        log_pred = model.predict(input_data)[0]
        prediction = np.expm1(log_pred)  # ✅ แปลงกลับจาก log เป็นราคาจริง

        current_year = datetime.datetime.now().year
        age = current_year - YearBuilt

        # ปรับราคาตามจำนวนโรงรถ
        if GarageCars == 0:
            prediction *= 0.85
        elif GarageCars == 1:
            prediction *= 0.93
        elif GarageCars == 3:
            prediction *= 1.08
        elif GarageCars == 4:
            prediction *= 1.15

        # ปรับราคาตามจำนวนห้องน้ำ
        if FullBath == 0:
            prediction *= 0.9
        elif FullBath == 1:
            prediction *= 0.95
        elif FullBath == 3:
            prediction *= 1.05
        elif FullBath >= 4:
            prediction *= 1.10

        # ปรับราคาตามอายุบ้าน
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

# -----------------------------
# ส่วนท้าย
# -----------------------------
st.markdown("---")
st.caption("🤖 พัฒนาโดย เสฎฐวุฒิ | วิชา AI | มข.")
