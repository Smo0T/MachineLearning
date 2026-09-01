import pandas as pd
from sklearn.ensemble import IsolationForest

# Данные: сумма транзакции и время суток
df = pd.DataFrame({'amount': [100, 200, 150, 100000, 300, 250, 50000], 
                   'hour': [10, 11, 12, 3, 14, 15, 4]})

model = IsolationForest(contamination=0.2, random_state=42)
df['is_anomaly'] = model.fit_predict(df) 
# -1: аномалия, 1: норма

print("Подозрительные транзакции:")
print(df[df['is_anomaly'] == -1])