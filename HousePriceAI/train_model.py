import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# ตรวจสอบว่าโฟลเดอร์ model มีหรือไม่ ถ้าไม่มีให้สร้าง
os.makedirs("model", exist_ok=True)

# โหลดข้อมูล
data = pd.read_csv("data/train.csv")

# เลือกฟีเจอร์หลัก
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = data[features]
y = data["SalePrice"]

# แบ่งข้อมูลเทรน/เทสต์
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและเทรนโมเดล
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ประเมินความแม่นยำ
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"🔹 MSE: {mse:.2f}")
print(f"🔹 R² Score: {r2:.4f}")

# บันทึกโมเดล
joblib.dump(model, "model/house_price_model.pkl")
print("💾 Model saved to model/house_price_model.pkl")
