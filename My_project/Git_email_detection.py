import pandas as pd

df=pd.read_csv('spam.csv')
df = df.sample(3000, random_state=42)  
print(df.head())
print("\nColumns:",df.columns)
print("\nshapes:",df.shape)

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def clean_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Split into words
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords + stem
    return ' '.join(words)
df['clean_body'] = df['body'].astype(str).apply(clean_text)

#vectorize text
y = df['label']
from sklearn.feature_extraction.text import CountVectorizer

# Initialize vectorizer
cv = CountVectorizer()

# Convert cleaned text to numerical data
X = cv.fit_transform(df['clean_body'])

print("Vectorized shape:", X.shape)  # (number of emails, number of unique words)

from sklearn.model_selection import train_test_split

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.naive_bayes import MultinomialNB

# Initialize and train
model = MultinomialNB()
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#display
# ----- Check a New Email -----

# Take input from the user
print("\nEnter a new email to check if it's spam:")
new_email = input("> ")

# Clean the email
cleaned_email = clean_text(new_email)

# Vectorize it
vectorized_email = cv.transform([cleaned_email])

# Predict using the trained model
prediction = model.predict(vectorized_email)

# Show result clearly
print("\n📩 Original Email:\n", new_email)
print("\n🧹 Cleaned Email:\n", cleaned_email)
print("\n📊 Prediction Result:")

if prediction[0] == 1:
    print("🚨 This email is SPAM!")
else:
    print("✅ This email is NOT SPAM.")

