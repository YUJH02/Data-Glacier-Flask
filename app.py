from flask import Flask, request, render_template
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

app = Flask(__name__)
prestige = sm.datasets.get_rdataset("Duncan", "carData").data
X = prestige[["income"]]
y = prestige["prestige"]

model = linear_model.LinearRegression(fit_intercept=True)
model.fit(X,y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    in_ = float(request.form.get("Income"))
    out_ = model.predict(np.array([in_]).reshape(-1, 1))[0]
    return render_template('index.html', prediction_text = f'Prestige is {out_}')

if __name__ == "__main__":
    app.run(port=5000)

