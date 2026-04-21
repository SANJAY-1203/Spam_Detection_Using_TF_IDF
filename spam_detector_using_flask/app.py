from flask import Flask, render_template, request
import pickle
from src.preprocessing import clean_text

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    result = "Spam ❌" if prediction == 1 else "Not Spam ✅"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
