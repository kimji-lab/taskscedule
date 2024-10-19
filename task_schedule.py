import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

start_date = datetime(2024, 10, 20)
end_date = datetime(2024, 11, 20)

task_names = ['Meeting', 'Bekerja', 'Belajar']

tasks = []
for i in range(1000):
    task_name = random.choice(task_names)
    duration = random.randint(1, 10)
    deadline = random_date(start_date, end_date).strftime('%Y-%m-%d')
    priority = random.randint(1, 3)
    tasks.append({'task_name': task_name, 'duration': duration, 'deadline': deadline, 'priority': priority})

df = pd.DataFrame(tasks)

df['deadline_days'] = (pd.to_datetime(df['deadline']) - start_date).dt.days

X = df[['duration', 'deadline_days']]
y = df['priority']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu')) 
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=1)

y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)

print(f"Akurasi Model: {accuracy * 100:.2f}%")

plt.plot(history.history['accuracy'])
plt.title('Akurasi Model per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
