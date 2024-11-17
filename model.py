import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
file_path = 'Supplementary data 1.xlsx'  # Replace with your local file path
data = pd.ExcelFile(file_path)
raw_data = data.parse('All Raw Data')

# Data cleaning
cleaned_data = raw_data.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
cleaned_data = cleaned_data.apply(pd.to_numeric, errors='coerce')

# Verify the data structure
print(cleaned_data.info())

# Check unique values in 'TYPE'
print("Unique values in TYPE column:", cleaned_data['TYPE'].unique())

# Value counts in the target variable
print("Value counts:\n", cleaned_data['TYPE'].value_counts())

# Impute missing values with the median
imputer = SimpleImputer(strategy='mean')
cleaned_data_imputed = pd.DataFrame(imputer.fit_transform(cleaned_data), columns=cleaned_data.columns)

# Separate features and target
X = cleaned_data_imputed.drop(['TYPE', 'SUBJECT_ID'], axis=1)
y = cleaned_data_imputed['TYPE']

# Select top 10 features based on ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 500],          # Number of trees
    'max_depth': [None, 10, 20, 30],              # Maximum depth of trees
    'min_samples_split': [2, 5, 10],              # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],                # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],       # Number of features to consider at each split
    'bootstrap': [True, False],                   # Whether bootstrap samples are used
}

# Use RandomizedSearchCV for efficiency
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=50,       # Number of combinations to try
    cv=5,            # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1        # Use all processors
)
# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print("Best parameters:", best_params)

# Train the model with the best parameters
best_model = random_search.best_estimator_
# Predictions on the test set
y_pred = best_model.predict(X_test)


import pickle

# Assuming `model` is your trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file) 


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('feature_selector', SelectKBest(score_func=f_classif, k=10)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(**best_params, random_state=42))
])

pipeline.fit(X_train, y_train)

# Save the pipeline
with open('pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# Use pipeline for predictions
y_pred = pipeline.predict(X_test)
print("Accuracy on test data:", accuracy_score(y_test, y_pred)*100)




