import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("iris.csv")
x = df.drop(columns="species")
y = df["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)

joblib.dump({"model": model, "scaler": scaler}, "model.pkl")
