# trainmodel.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

# -----------------------------
# ตรวจสอบโฟลเดอร์
# -----------------------------
os.makedirs("model", exist_ok=True)

# -----------------------------
# โหลดข้อมูล
# -----------------------------
data = pd.read_csv("data/train.csv")

# ✅ กรองข้อมูล outlier (บ้านที่ใหญ่มากแต่ราคาต่ำผิดปกติ)
data = data[data["GrLivArea"] < 4500]

# เลือกฟีเจอร์สำคัญ
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = data[features]
y = data["SalePrice"]

# ✅ ทำ log transform เพื่อลดความเอียงของราคา
y = np.log1p(y)

# -----------------------------
# แบ่งข้อมูล train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# เทรนโมเดล Random Forest
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,       # จำนวนต้นไม้
    max_depth=None,         # ให้โมเดลเรียนรู้ได้เต็มที่
    random_state=42,
    n_jobs=-1               # ใช้ทุกคอร์ CPU เร่งความเร็ว
)
model.fit(X_train, y_train)

# -----------------------------
# ประเมินผล
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"🔹 MSE: {mse:.4f}")
print(f"🔹 R² Score: {r2:.4f}")

# -----------------------------
# บันทึกโมเดล
# -----------------------------
joblib.dump(model, "model/best_model.pkl")
print("💾 Model saved as model/best_model.pkl")
