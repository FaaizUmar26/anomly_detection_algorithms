from sklearn.svm import SVC

def train_svm_classifier(X_train, y_train, kernel='linear'):
    svm_classifier = SVC(kernel=kernel)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier
