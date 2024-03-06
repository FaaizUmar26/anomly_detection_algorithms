import tensorflow as tf

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(11,)),#flatten layer transform multi-dimensional data to one dimensional arrayu
        tf.keras.layers.Dense(128, activation=tf.nn.relu),#dense layer is used for learning complex pattern
        tf.keras.layers.Dense(128, activation=tf.nn.relu),#Activation functions help in capturing complex relationships between inputs and outputs,
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)#sigmoid is used to give value 0,1
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model#the main aim of the optimizer is to train the model perfectlly so there is no chance of losss

def train_model(model, X_train, y_train, epochs=20):
    history = model.fit(X_train, y_train, epochs=epochs)
    return history
