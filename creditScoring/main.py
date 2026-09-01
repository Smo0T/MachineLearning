import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Загрузка
df = pd.read_csv('creditScoring/credit_data.csv')
X, y = df.drop('default', axis=1), df['default']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# На эту:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
pipeline.fit(X_train, y_train)

auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {auc:.2f}")