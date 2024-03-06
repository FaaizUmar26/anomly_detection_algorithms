def get_model_accuracy(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    return test_accuracy
