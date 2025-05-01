import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


data = {
    'age': [5, 7, 3, 6, 2, 4, 8, 1],
    'temperature': [30, 25, 15, 20, 10, 12, 28, 8],
    'likes_ice_cream': [1, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)
print(df)

X = df[['age', 'temperature']]   # תכונות
y = df['likes_ice_cream']        # מטרה

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("דיוק (Accuracy):", accuracy)
print("דיוק חיובי (Precision):", precision)
print("רגישות (Recall):", recall)
print("מדד F1:", f1)

print("תחזיות המודל:", y_pred.tolist())
print("התשובות האמיתיות:", y_test.tolist())

# בניית טבלת Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# הצגת הטבלה
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Greens)  # אפשר גם cmap אחר לצבעים שונים
plt.show()

cm = np.array([[3, 9], [0, 2]])
y_test = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
y_pred = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("דיוק (Accuracy):", accuracy)
print("דיוק חיובי (Precision):", precision)
print("רגישות (Recall):", recall)
print("מדד F1:", f1)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()