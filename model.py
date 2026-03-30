import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# download stopwords
nltk.download('stopwords')

print("🚀 STEP 1: Loading dataset...")

# load dataset (keep small for speed)
df = pd.read_csv("data/fake_job_postings.csv").head(200)

print("✅ STEP 2: Dataset loaded")

# use description column
df['text'] = df['description'].fillna('')

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

print("🧹 STEP 3: Cleaning text...")
df['text'] = df['text'].apply(clean_text)

print("✅ STEP 4: Text cleaned")

# features & labels
X = df['text']
y = df['fraudulent']

print("⚙ STEP 5: Vectorizing...")
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(X)

print("✅ STEP 6: Vectorization done")

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("🤖 STEP 7: Training model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

print("🎉 MODEL READY SUCCESSFULLY!")