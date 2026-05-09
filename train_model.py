# 导入库
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import joblib
from pathlib import Path


# 加载数据
data = pd.read_csv("training_data.csv")
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 训练模型
model = Pipeline(
    [("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())]
)

model.fit(x, y)

# 保存模型
joblib.dump(model, "model.pkl")

# 模型到文件
lr = model.named_steps["linear"]
Path("linear_model.txt").write_text(
    f"Coefficients: {lr.coef_}\nIntercept: {lr.intercept_}\n", encoding="utf-8"
)
