import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Загрузка данных
try:
    df = pd.read_csv('realEstansePrice/apartments.csv')
except FileNotFoundError:
    print("Ошибка: Файл не найден. Проверьте путь.")
    exit()

# 2. Анализ данных (чтобы понимать, что пришло)
print(f"--- Загружено данных: {len(df)} строк ---")
print("Первые 5 строк:")
print(df.head())
print("\nСтатистика цен:")
print(df['price'].describe())

# 3. Предобработка
# Удаляем выбросы
original_len = len(df)
df = df[df['price'] < df['price'].quantile(0.95)]
print(f"\nУдалено выбросов: {original_len - len(df)}")

# 4. Разделение для проверки качества
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Обучение
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Оценка
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Средняя ошибка предсказания (MAE): {mae:,.0f} руб.\nКоэффициент детерминации (R²): {r2:.2f}")