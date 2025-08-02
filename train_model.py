import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils.text_cleaner import clean_text
import os

# Load true and fake datasets
true_df = pd.read_csv('data/true.csv')
fake_df = pd.read_csv('data/fake.csv')

# Assign labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Make sure the text column exists (adjust if your column name is different)
# For example, if your news content is in a column named 'text' or 'content':
# Replace 'text' below with your actual column name.

df['text'] = df['text'].astype(str).apply(clean_text)

X = df['text']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create model directory if not exists
os.makedirs('model', exist_ok=True)

# Save model and vectorizer
with open('model/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully!")

