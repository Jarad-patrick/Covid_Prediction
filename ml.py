import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np


df = pd.read_csv('covid_dataset.csv')

X = df.drop('Has_Covid', axis=1)
y = df['Has_Covid'].map({'No': 0, 'Yes': 1})


# Column groups
numerical_cols = ['Age', 'Fever']
categorical_cols = ['Gender', 'Cough', 'City']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

#pipliene
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]
)

# Train model
model.fit(X_train, y_train)


#tuning recall by reducing threshold to 0.3
y_proba = model.predict_proba(X_test)[:, 1]

threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)


cm = confusion_matrix(y_test, y_pred_custom)
cr = classification_report(y_test, y_pred_custom)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)
