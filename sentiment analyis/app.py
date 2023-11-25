from flask import Flask, render_template, request

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Function to preprocess a text string
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

# Load your data
df = pd.read_csv("EcoPreprocessed.csv")

# Preprocess the 'review' column
df['sentence'] = df['review'].apply(preprocess_text)

# Split the data into training and testing sets
X = df['sentence']
y = df['division']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Load the TF-IDF vectorizer and fit on the training data
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_vectorizer.fit(X_train)

# Load the trained Random Forest model
model_filename = 'sentiment_analysis_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    
    # Preprocess the user input
    preprocessed_input = preprocess_text(user_input)
    
    # Transform the preprocessed input using the TF-IDF vectorizer
    input_vectorized = tfidf_vectorizer.transform([preprocessed_input])
    
    # Make predictions using the model
    prediction = model.predict(input_vectorized)
    
    # Map numeric predictions back to text labels
    label_mapping_reverse = {1: 'positive', 0: 'neutral', -1: 'negative'}
    predicted_sentiment = label_mapping_reverse[prediction[0]]
    
    return render_template('index.html', user_input=user_input, sentiment=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
