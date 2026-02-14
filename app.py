from flask import Flask, render_template
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

def generate_data(n=100):
    data = {
        "solar": [random.uniform(1,4) for _ in range(n)],
        "wind": [random.uniform(0,4) for _ in range(n)],
        "hydro": [random.uniform(0,3) for _ in range(n)],
        "consumption": [random.uniform(1,5) for _ in range(n)]
    }
    return pd.DataFrame(data)

def analyze(df):
    return {"production": df[["solar","wind","hydro"]].sum().sum(),
            "consumption": df["consumption"].sum()}

EMISSION_FACTOR = 0.92
def carbon_saved(df):
    renewable = df[["solar","wind","hydro"]].sum().sum()
    return round(renewable * EMISSION_FACTOR, 2)

def suggestion(df):
    extra = df[["solar","wind","hydro"]].sum(axis=1).sum() - df["consumption"].sum()
    if extra > 0:
        return "Store Energy in Battery"
    elif extra <0:
        return "Reduce Consumption"
    else:
        return "Balanced System"

def predict(df):
    X = np.arange(len(df)).reshape(-1,1)
    y = df["consumption"]
    model = LinearRegression()
    model.fit(X, y)
    future = model.predict([[len(df)+5]])
    return round(future[0],2)

@app.route("/")
def home():
    df = generate_data()
    stats = analyze(df)
    carbon = carbon_saved(df)
    sugg = suggestion(df)
    prediction = predict(df)

    return render_template("dashboard.html",
                           production=round(stats["production"],2),
                           consumption=round(stats["consumption"],2),
                           carbon=carbon,
                           suggestion=sugg,
                           prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)