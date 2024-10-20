import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data import data  # وارد کردن داده‌ها از فایل data.py

# تبدیل داده‌ها به DataFrame
df = pd.DataFrame(data)

# تقسیم داده‌ها به مجموعه آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# تبدیل متون به ویژگی‌ها با استفاده از TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ایجاد و آموزش مدل
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# پیش‌بینی بر روی داده‌های تست
y_pred = model.predict(X_test_vec)

# محاسبه دقت مدل
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# دریافت ورودی از کاربر و پیش‌بینی
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break
    
    # تبدیل ورودی به ویژگی‌ها
    user_input_vec = vectorizer.transform([user_input])
    
    # پیش‌بینی احساسات
    prediction = model.predict(user_input_vec)
    
    print(f'Sentiment: {prediction[0]}')
