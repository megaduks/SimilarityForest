import pandas as pd
import numpy as np

from numpy.linalg import norm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from DataParser import DataParser

from simforest import SimilarityForest


def cosine(a,b):
    return np.dot(a,b)/(norm(a)*norm(b))


if __name__ == '__main__':

    y, X = DataParser("data/a1a.txt", 123).parse()
    # y, X = DataParser("data/breast-cancer.txt", 10).parse()
    # y, X = DataParser("data/german-numer.txt", 24).parse()
    # y, X = DataParser("data/heart.txt", 13).parse()
    # y, X = DataParser("data/ionosphere_scale.txt", 34).parse()
    # y, X = DataParser("data/mushrooms.txt", 112).parse()
    # y, X = DataParser("data/splice.txt", 60).parse()

    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns).to_numpy()
    y = y.apply(pd.to_numeric).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    sim_forest = SimilarityForest(n_estimators=10, similarity_function=cosine)
    sim_forest.fit(X_train, y_train)
    sf_pred = sim_forest.predict(X_test)

    print(f"Similarity forest: {accuracy_score(y_test, sf_pred)}\n")
    print(confusion_matrix(y_test, sf_pred))
    print(classification_report(y_test, sf_pred))

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X_train, y_train.ravel())
    rf_pred = rf.predict(X_test)

    print(f"Random forest: {accuracy_score(y_test, rf_pred)}\n")
    print(confusion_matrix(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))
