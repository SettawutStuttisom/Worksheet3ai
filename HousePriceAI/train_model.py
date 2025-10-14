# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

# -----------------------------
# ตรวจสอบโฟลเดอร์
# -----------------------------
os.makedirs("model", exist_ok=True)

# -----------------------------
# โหลดข้อมูล
# -----------------------------
data = pd.read_csv("data/train.csv")

# -----------------------------
# เลือกฟีเจอร์หลัก
# -----------------------------
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = data[features]
y = np.log1p(data["SalePrice"])  # ✅ แปลงเป็น log เพื่อให้โมเดลเรียนรู้ได้ดีขึ้น

# -----------------------------
# แบ่งข้อมูล train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# สร้างและเทรนโมเดล
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -----------------------------
# ประเมินโมเดล
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

print("✅ Model trained successfully!")
print(f"🔹 MSE: {mse:.2f}")
print(f"🔹 R² Score: {r2:.4f}")

# -----------------------------
# บันทึกโมเดล
# -----------------------------
joblib.dump(model, "model/best_model.pkl")
print("💾 Model saved to model/best_model.pkl")
