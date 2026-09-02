import pandas as pd # Импортируем библиотеку Pandas для работы с данными в виде таблиц (DataFrames).
from sklearn.ensemble import RandomForestRegressor # Импортируем алгоритм "Случайный лес" для задач регрессии (предсказание числового значения, в данном случае — цены).
from sklearn.metrics import mean_absolute_error, r2_score # Импортируем метрики оценки качества: MAE и R^2.
from sklearn.model_selection import train_test_split # Импортируем функцию для разделения данных на обучающую и тестовую выборки.

df = pd.read_csv('realEstansePrice/apartments.csv') # Загружаем данные из CSV в память

# Анализ данных
print(f"Загружено данных: {len(df)} строк") # Общее количество квартир
print(f"\nПервые 5 строк: \n{df.head()}")
print(f"\nСтатистика цен:\n{df['price'].describe()}")

if 'area' in df.columns and df['area'].isnull().any():
    median_area = df['area'].median()
    df['area'] = df['area'].fillna(median_area)

# Предобработка данных
# Удаляем выбросы (очень дорогие квартиры, которые могут исказить обучение)
original_len = len(df)
df = df[df['price'] < df['price'].quantile(0.95)] # Удаляем самые дорогие 5% квартир.
print(f"\nУдалено выбросов (самых дорогих квартир): {original_len - len(df)}")

# Разделение данных на обучающую и тестовую выборки
X = df.drop('price', axis=1) # X (признаки) — это все колонки, кроме 'price'. axis=1 означает, что мы удаляем колонку.
y = df['price'] # y (целевая переменная) — это колонка 'price', которую мы хотим предсказать.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, # 20% данных будет использоваться для тестирования модели.
    random_state=42, # Фиксирует "случайность" разбиения. При одинаковом random_state разбиение всегда будет одинаковым.
)

# Обучение модели
# n_estimators=100: Количество деревьев в лесу. Больше деревьев обычно дает лучшую точность, но замедляет обучение.
# random_state=42: Гарантирует, что при каждом запуске модели будут строиться одинаковые деревья.
model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)

# Оценка качества модели
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions) # Рассчитываем MAE (Mean Absolute Error). Это средняя абсолютная разница между реальными ценами (y_test) и предсказанными (predictions). 
r2 = r2_score(y_test, predictions) # Рассчитываем R² (Coefficient of Determination). Этот показатель говорит, какая доля вариации целевой переменной (цены) объясняется моделью.
                                  # R² = 1 означает идеальное предсказание. R² = 0 означает, что модель предсказывает так же плохо, как если бы мы просто брали среднюю цену.

print(f"Средняя ошибка предсказания (MAE): {mae:,.0f} руб.\nКоэффициент детерминации (R²): {r2:.2f}")

results = pd.DataFrame({
    'Реальная цена': y_test,
    'Предсказание': predictions,
    'Ошибка (абс)': abs(y_test - predictions)
})

print("\n--- Сравнение первых 5 предсказаний ---")
print(results.head())

results['Ошибка (%)'] = (results['Ошибка (абс)'] / results['Реальная цена']) * 100
print(f"\nСредняя ошибка в процентах: {results['Ошибка (%)'].mean():.2f}%")