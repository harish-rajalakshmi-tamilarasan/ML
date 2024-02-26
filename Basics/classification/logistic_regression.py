from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import Basics.classification.load_classification_data as load_classification_data 

X_train, X_test, y_train, y_test = load_classification_data.get_data()

model = LogisticRegression(random_state=42)

param_grid = {
    'C': [0.1,1,10,100],
    'penalty': ['l1','l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", verbose =2)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
best_model.fit(X_train,y_train)
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
