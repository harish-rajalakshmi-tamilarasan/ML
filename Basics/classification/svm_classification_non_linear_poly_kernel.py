from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import Basics.classification.load_classification_data as load_classification_data  

X_train, X_test, y_train, y_test = load_classification_data.get_data()

model = make_pipeline(StandardScaler(),SVC(kernel="poly",coef0=1,degree=3, C=5))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))