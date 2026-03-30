from flask import Flask, render_template, request
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# import FULL model file
import model

# download stopwords
nltk.download('stopwords')

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z ]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# home page
@app.route('/')
def home():
    return render_template('index.html')

# prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['job_text']
    cleaned = clean_text(text)

    vector = model.vectorizer.transform([cleaned])
    prediction = model.model.predict(vector)

    result = "⚠ Fake Job" if prediction[0] == 1 else "✅ Real Job"

    return render_template('index.html', prediction_text=result)

# run app
if __name__ == "__main__":
    app.run(debug=True)