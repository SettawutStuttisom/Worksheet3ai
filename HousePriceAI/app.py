import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
import os
import sys

st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 ระบบทำนายราคาบ้านด้วย AI (อัปเดตอัตโนมัติ)")
st.write("โมเดล Random Forest พร้อมระบบโหลด/เทรนอัตโนมัติใน Streamlit")

# -----------------------------
# 🔹 ตรวจสอบและโหลดโมเดล
# -----------------------------
MODEL_PATH = "model/best_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ ไม่พบโมเดลในระบบ กำลังเทรนใหม่อัตโนมัติ...")

    # ตรวจว่ามีไฟล์ train_model.py อยู่ไหม
    if os.path.exists("train_model.py"):
        try:
            # ใช้ exec แทน subprocess (ปลอดภัยกว่าใน Streamlit)
            with open("train_model.py", "r", encoding="utf-8") as f:
                code = compile(f.read(), "train_model.py", "exec")
                exec(code, globals())
            st.success("✅ เทรนโมเดลใหม่สำเร็จ!")
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการสร้างโมเดล: {e}")
            st.stop()
    else:
        st.error("❌ ไม่พบไฟล์ train_model.py")
        st.stop()

# โหลดโมเดลที่เทรนแล้ว
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

# -----------------------------
# เตรียมข้อมูลสำหรับโมเดล
# -----------------------------
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
        prediction = np.expm1(log_pred)  # แปลง log กลับเป็นราคาจริง

        current_year = datetime.datetime.now().year
        age = current_year - YearBuilt

        # -----------------------------
        # ปรับราคาเพิ่มเติม (Fine-tuning)
        # -----------------------------
        # โรงรถ
        garage_factor = {0: 0.85, 1: 0.93, 2: 1.00, 3: 1.08, 4: 1.15}
        prediction *= garage_factor.get(GarageCars, 1.0)

        # ห้องน้ำ
        bath_factor = {0: 0.9, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.10}
        prediction *= bath_factor.get(FullBath, 1.0)

        # อายุบ้าน
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
