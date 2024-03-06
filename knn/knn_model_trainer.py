from sklearn.neighbors import KNeighborsClassifier

def train_knn_classifier(X_train, y_train, k=5):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    return knn_classifier
