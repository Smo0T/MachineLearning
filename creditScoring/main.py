import pandas as pd # Библиотека для работы с таблицами (DataFrames)
from sklearn.ensemble import RandomForestClassifier # Алгоритм "Случайный лес" для классификации
from sklearn.pipeline import Pipeline # Инструмент для объединения шагов обработки и модели
from sklearn.preprocessing import StandardScaler # Преобразователь: приводит все числа к среднему 0 и дисперсии 1
from sklearn.model_selection import train_test_split # Функция для деления данных на обучающую и тестовую части
from sklearn.metrics import roc_auc_score, classification_report # Метрики оценки качества

df = pd.read_csv('creditScoring/credit_data.csv') # Загружаем данные из CSV в память
X, y = df.drop('default', axis=1), df['default'] # Разделяем: X — признаки (факторы), y — целевая переменная (результат)

# 2. Анализ данных
print("Анализ данных для кредитного скоринга")
print(f"Загружено {len(df)} строк с БД.")
print(f"\nПервые 5 строк: \n{df.head()}")
print("\nСтатистика по доходам (income):")
print(df['income'].describe())

if 'income' in df.columns and df['income'].isnull().any():
    median_income = df['income'].median()
    df['income'] = df['income'].fillna(median_income)
else:
    print("\nКолонка 'income' не содержит пропусков. (данные заполнены)")

# Создаем конвейер (Pipeline)
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Масштабируем признаки (чтобы доход 50000 не перевешивал возраст 30)
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42)) 
    # Модель. class_weight='balanced' заставляет модель больше "ценить" редкий класс (дефолт)
])

# Делим данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% данных уходит на проверку модели
    random_state=42, # Число для фиксации случайности (чтобы результат был повторяем)
    stratify=y # Гарантирует, что в тесте и трейне доля "дефолтчиков" одинаковая
)

print(f"Размер тестовой выборки: {len(y_test)} строк")
print(f"Распределение классов в тесте: {y_test.value_counts().to_dict()}")

pipeline.fit(X_train, y_train) # Обучаем весь Pipeline на обучающих данных

y_prob = pipeline.predict_proba(X_test)[:, 1] # Получаем вероятности (0.0 - 1.0) для каждого теста
auc = roc_auc_score(y_test, y_prob) # Считаем ROC-AUC — качество ранжирования клиентов по риску
print(f"\nROC-AUC Score: {auc:.2f}")

y_pred = pipeline.predict(X_test) # Получаем жесткие прогнозы (0 или 1) для отчета
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred)) # Печатаем Precision, Recall, F1 для обоих классов

feature_importances = pd.Series(pipeline.named_steps['clf'].feature_importances_, index=X.columns)
print("\nВажность признаков:")
print(feature_importances.sort_values(ascending=False))