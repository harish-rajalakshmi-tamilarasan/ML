from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target


def get_data():
    return train_test_split(X, y, random_state=42)
