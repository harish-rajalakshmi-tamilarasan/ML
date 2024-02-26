from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import load_classification_data 

X_train, X_test, y_train, y_test = load_classification_data.get_data()

model = VotingClassifier(
    estimators=[
        ("lr",LogisticRegression(random_state=42)),
        ('rf',RandomForestClassifier(random_state=42)),
        ('svc',SVC(kernel="rbf",random_state=42)) 
                ]
    )
#for soft voting set
model.voting = "soft"
model.named_estimators_["svc"].probability = True
model.fit(X_train,y_train)
for name, clf in model.named_estimators_.items():
    print(name, "=", clf.score(X_test,y_test))

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



