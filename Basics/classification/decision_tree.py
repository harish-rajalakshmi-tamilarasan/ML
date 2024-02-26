from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from graphviz import Source
import load_classification_data 

X_train, X_test, y_train, y_test = load_classification_data.get_data()

model = DecisionTreeClassifier(random_state=42)
param_grid = {
   # 'criterion': ['gini'],
    'max_depth': range(8, 15),
    #'min_samples_split': range(2, 10),
    #'min_samples_leaf': range(1, 5)
}

grid_Search = GridSearchCV(model,param_grid=param_grid, cv=5, scoring='accuracy',n_jobs=1)
grid_Search.fit(X_train,y_train)
best_model = grid_Search.best_estimator_
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

export_graphviz(best_model,out_file="mnist.dot")
Source.from_file("mnist.dot")
