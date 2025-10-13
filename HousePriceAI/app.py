import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# โหลดข้อมูล
data = pd.read_csv("data/train.csv")

# เลือกฟีเจอร์ที่เกี่ยวข้องกับราคาบ้าน
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = data[features]
y = data["SalePrice"]

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Decision Tree ----------------
dt_params = {'max_depth': [3, 5, 10, None]}
dt = DecisionTreeRegressor(random_state=42)
dt_gs = GridSearchCV(dt, dt_params, cv=3, scoring='r2', n_jobs=-1)
dt_gs.fit(X_train, y_train)

best_dt = dt_gs.best_estimator_
y_pred_dt = best_dt.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)
rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)

# ---------------- Neural Network (ANN) ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_params = {'hidden_layer_sizes': [(64,), (128,), (128, 64)], 'alpha': [0.0001, 0.001]}
mlp = MLPRegressor(max_iter=1000, random_state=42)
mlp_gs = GridSearchCV(mlp, mlp_params, cv=3, scoring='r2', n_jobs=-1)
mlp_gs.fit(X_train_scaled, y_train)

best_mlp = mlp_gs.best_estimator_
y_pred_mlp = best_mlp.predict(X_test_scaled)
r2_mlp = r2_score(y_test, y_pred_mlp)
rmse_mlp = mean_squared_error(y_test, y_pred_mlp, squared=False)

# ---------------- ผลลัพธ์ ----------------
print("📊 ผลลัพธ์การเปรียบเทียบโมเดล\n")
print(f"Decision Tree: R2 = {r2_dt:.3f}, RMSE = {rmse_dt:.2f}, Params = {dt_gs.best_params_}")
print(f"Neural Network: R2 = {r2_mlp:.3f}, RMSE = {rmse_mlp:.2f}, Params = {mlp_gs.best_params_}")

# ---------------- เลือกโมเดลที่ดีกว่า ----------------
if r2_mlp > r2_dt:
    best_model = best_mlp
    model_name = "Neural Network"
else:
    best_model = best_dt
    model_name = "Decision Tree"

# บันทึกโมเดล
joblib.dump(best_model, "best_model.pkl")
print(f"\n✅ เลือกใช้โมเดล: {model_name}")
print("💾 โมเดลถูกบันทึกในไฟล์ best_model.pkl แล้ว")
