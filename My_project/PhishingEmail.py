import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#########################################################################################################

# Load dataset (you can replace this with your own dataset)
# Sample dataset structure should have 'text' and 'label' columns (1 for phishing, 0 for legitimate)
# For this example, we'll create a sample dataframe
# data = {
#     'text': [
#         "Urgent: Your account will be suspended unless you verify your details immediately!",
#         "Hi John, just checking in about our meeting tomorrow at 2pm.",
#         "Click this link to claim your free prize: http://malicious.site/offer",
#         "Your monthly invoice is ready. Please find attached.",
#         "Your bank account has been compromised. Verify your identity now!",
#         "Reminder: Your appointment with Dr. Smith is scheduled for next week."
#     
#     'label': [1, 0, 1, 0, 1, 0]
# }
df=pd.read_csv('spam.csv')

df = pd.DataFrame(data)

# If you have a CSV file, use:
# df = pd.read_csv('phishing_emails.csv')

print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())


##########################################################################


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)
print("\nSample preprocessed texts:")
print(df[['text', 'processed_text']].head())



################################################################################

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['processed_text'])
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")



##################################################################################

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'], 
                yticklabels=['Legitimate', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return model

# Logistic Regression
print("\nLogistic Regression:")
lr_model = LogisticRegression(max_iter=1000)
train_and_evaluate(lr_model, X_train, y_train, X_test, y_test)

# Random Forest
print("\nRandom Forest:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_evaluate(rf_model, X_train, y_train, X_test, y_test)

# Support Vector Machine
print("\nSupport Vector Machine:")
svm_model = SVC(kernel='linear', probability=True)
train_and_evaluate(svm_model, X_train, y_train, X_test, y_test)




#######################################################################################33333



# Example for Logistic Regression tuning
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("\nBest parameters for Logistic Regression:")
print(grid_search.best_params_)

best_lr_model = grid_search.best_estimator_
train_and_evaluate(best_lr_model, X_train, y_train, X_test, y_test)


#########################################################################################
# Create a pipeline that includes preprocessing and model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(C=10, penalty='l2', solver='liblinear', max_iter=1000))
])

# Fit the pipeline on the entire dataset
pipeline.fit(df['processed_text'], df['label'])

# Function to predict new emails
def predict_email(email_text):
    processed_text = preprocess_text(email_text)
    prob = pipeline.predict_proba([processed_text])[0]
    prediction = pipeline.predict([processed_text])[0]
    
    print(f"\nEmail: {email_text}")
    print(f"Processed: {processed_text}")
    print(f"Prediction: {'Phishing' if prediction == 1 else 'Legitimate'}")
    print(f"Confidence: {max(prob)*100:.2f}%")
    return prediction

# Test with new emails
test_emails = [
    "Your PayPal account needs verification. Click here to update your details now!",
    "Meeting reminder for tomorrow at 3pm in conference room B",
    "Congratulations! You've won an iPhone. Claim your prize by visiting this link"
]

for email in test_emails:
    predict_email(email)
    
    
    
    ###################################################################
    
    import joblib

# Save the model
joblib.dump(pipeline, 'phishing_email_detector.pkl')

# Load the model (in another script)
# loaded_model = joblib.load('phishing_email_detector.pkl')

######################################################################

# Example of URL detection feature
def contains_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return 1 if url_pattern.search(text) else 0

df['contains_url'] = df['text'].apply(contains_url)

# You can add this feature to your model by creating additional feature columns