import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1_Score:", f1_score(y_test, y_pred))
# Count each class
class_counts = df['label'].value_counts()

# Plot it
plt.figure(figsize=(6,4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title("Distribution of Ham vs Spam Messages")
plt.xlabel("Message Type (0 = Ham, 1 = Spam)")
plt.ylabel("Number of Messages")
plt.xticks([0, 1], ['Ham (0)', 'Spam (1)'])
plt.tight_layout()
plt.show()

