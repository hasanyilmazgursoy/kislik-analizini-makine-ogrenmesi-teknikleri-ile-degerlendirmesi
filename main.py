# Gerekli kütüphaneler
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Veri setini okuma
data = pd.read_csv('data.csv')

# Veri seti hakkında bilgi
print(data.head())  # İlk 5 satır
print(data.info())  # Veri seti hakkında bilgi
print(data.describe())  # İstatistiksel bilgiler
print(data.columns)  # Sütunlar

# Veri setindeki eksik verileri kontrol etme
print(data.isnull().sum())  # Her sütundaki eksik veri sayısı

# Kategori sütunlarını sayısal değerlere dönüştürme
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)
data = pd.get_dummies(data, columns=['Interest'], drop_first=True)

# Kişilik sütunlarını One-Hot Encoding ile dönüştürme
data = pd.get_dummies(data, columns=['Personality'], prefix='Personality', drop_first=True)

# Dönüşüm sonrası veri setini inceleme
print(data.head())

# Veri Normalizasyonu
scaler = MinMaxScaler()
numeric_columns = ['Age', 'Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Feature Engineering
data['Age Group'] = pd.cut(data['Age'], bins=[18, 25, 35, 45, 60], labels=['18-25', '25-35', '35-45', '45-60'])
data = pd.get_dummies(data, columns=['Age Group'], drop_first=True)

# Eğitim ve Test Setlerine Ayırma
X = data.drop(columns=[col for col in data.columns if 'Personality' in col])
y = data['Personality_ENFP']  # Örnek olarak ENFP kişilik türünü hedef alıyoruz

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modellerin tanımlanması ve eğitilmesi
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC()
}

# Sonuçları saklamak için bir liste oluşturma
results = []

# Her model için tahminler ve değerlendirmeler
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Sonuçları değerlendirme
    accuracy = accuracy_score(y_test, predictions) * 100
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    # Sonuçları listeye ekleme
    results.append({
        "Model": model_name,
        "Accuracy (%)": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    })

    # Karışıklık matrisini görselleştirme
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Diğer', 'ENFP'],
                yticklabels=['Diğer', 'ENFP'])
    plt.title(f'Karışıklık Matrisi: {model_name}', fontsize=16)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.show()

# Sonuçları DataFrame olarak gösterme
results_df = pd.DataFrame(results)
print("\n### Modellerin Sonuçları ###")
print(results_df)
