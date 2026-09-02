import pandas as pd # Библиотека для работы с таблицами (DataFrames)
from sklearn.ensemble import IsolationForest # Импортируем алгоритм Isolation Forest из библиотеки Scikit-learn. 

# Создаем DataFrame с данными о транзакциях
df = pd.DataFrame({
    'amount': [100, 200, 150, 100000, 300, 250, 50000, 180, 220, 130, 90000, 400, 280, 60000, 160, 350, 210, 70000], 
    'hour': [10, 11, 12, 3, 14, 15, 4, 10, 11, 12, 3, 14, 15, 4, 10, 11, 12, 3]
})

# Инициализируем модель Isolation Forest:
# 'contamination': Это параметр, который указывает модели, какую долю данных она ожидает считать аномалиями.
# 'random_state=42': Фиксирует случайность. Позволяет получать одни и те же результаты при каждом запуске скрипта.
model = IsolationForest(contamination=0.2, random_state=42)

# Обучение модели
df['is_anomaly'] = model.fit_predict(df)

print("Подозрительные транзакции:")
print(df[df['is_anomaly'] == -1])

print("\nОбычные транзакции:")
print(df[df['is_anomaly'] == 1].head())