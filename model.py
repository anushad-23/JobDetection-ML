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

print("STEP 7: Training model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

print("🎉 MODEL READY SUCCESSFULLY!")

# VISUALIZATION DATA - NEW SECTION
print("📊 Computing visualization data...")

# Full dataset for stats (no head limit)
full_df = pd.read_csv("data/fake_job_postings.csv")
full_df['text'] = full_df['description'].fillna('').apply(clean_text)
fraud_count = (full_df['fraudulent'] == 1).sum()
real_count = len(full_df) - fraud_count
dataset_stats = {'fraud': int(fraud_count), 'real': int(real_count), 'total': len(full_df)}

# Test predictions & metrics
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
test_accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()

# Feature importance (top 10)
importances = model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
top_features = sorted(zip(importances, feature_names), reverse=True)[:10]
feature_importance = [{'feature': name, 'importance': imp} for imp, name in top_features]

# Fraud wordcloud base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
fraud_texts = full_df[full_df['fraudulent'] == 1]['text'].str.cat(sep=' ')
if fraud_texts.strip():
    wc = WordCloud(width=400, height=200, background_color='white').generate(fraud_texts)
    img = io.BytesIO()
    plt.figure(figsize=(5, 2.5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='PNG', bbox_inches='tight', pad_inches=0)
    img.seek(0)
    fraud_wordcloud_b64 = base64.b64encode(img.read()).decode()
else:
    fraud_wordcloud_b64 = ''

# Metrics dict
metrics = {
    'accuracy': round(test_accuracy, 3),
    'precision_0': round(class_report['0']['precision'], 3),
    'precision_1': round(class_report['1']['precision'], 3),
    'recall_0': round(class_report['0']['recall'], 3),
    'recall_1': round(class_report['1']['recall'], 3),
    'f1_0': round(class_report['0']['f1-score'], 3),
    'f1_1': round(class_report['1']['f1-score'], 3),
    'conf_matrix': conf_matrix,
    'feature_importance': feature_importance,
    'fraud_wordcloud_b64': fraud_wordcloud_b64,
    'dataset_stats': dataset_stats
}

print("✅ Visualization data ready!")
