import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping




class BatchLossLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

def build_model(activation, l2_lambda=None, dropout_rate=None):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    
    #First hidden layer
    if l2_lambda:
        model.add(Dense(500, activation=activation, kernel_regularizer=l2(l2_lambda)))
    else:
        model.add(Dense(500, activation=activation))
    
    #Dropout
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    
    #Second hidden layer
    if l2_lambda:
        model.add(Dense(500, activation=activation, kernel_regularizer=l2(l2_lambda)))
    else:
        model.add(Dense(500, activation=activation))
    
    
    if dropout_rate:
        model.add(Dropout(dropout_rate))
    
   
    model.add(Dense(10, activation='softmax'))
    return model

def train_model(model, x_train, y_train, early_stopping=False):
    #Compile
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True))
    batch_logger = BatchLossLogger()
    callbacks.append(batch_logger)
    
    #Train model
    history = model.fit(x_train, y_train,
                        epochs=250,
                        batch_size=1000,
                        validation_split=0.4,
                        callbacks=callbacks,
                        verbose=0)
    return history, batch_logger.batch_losses

def plot_batch_loss(results):
    plt.figure(figsize=(14, 6))
    for name, result in results.items():
        plt.plot(result["batch_losses"], label=name)
    plt.title('Batch Loss During Training', fontsize=14)
    plt.xlabel('Batch Number', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

def plot_validation_error(results):
    plt.figure(figsize=(14, 6))
    for name, result in results.items():
        val_error = [1 - acc for acc in result["history"].history['val_accuracy']]
        plt.plot(val_error, label=name)
    plt.title('Validation Error', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.ylim(0, 1)  
    plt.xticks([0, 50, 100, 150, 200, 250]) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

def plot_results(results):
    #Create figure
    plt.figure(figsize=(14, 10))

    #Classification Error
    plt.subplot(2, 1, 1)
    for name, result in results.items():
        val_error = [1 - acc for acc in result["history"].history['val_accuracy']]
        plt.plot(val_error, label=f'{name} Validation')
        train_error = [1 - acc for acc in result["history"].history['accuracy']]
        plt.plot(train_error, label=f'{name} Training', linestyle='--')
    plt.title('Classification Error', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.ylim(0, 1) 
    plt.xticks([0, 50, 100, 150, 200, 250]) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)

    #Loss
    plt.subplot(2, 1, 2)
    for name, result in results.items():
        val_loss = result["history"].history['val_loss']
        plt.plot(val_loss, label=f'{name} Validation')
        train_loss = result["history"].history['loss']
        plt.plot(train_loss, label=f'{name} Training', linestyle='--')
    plt.title('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks([0, 50, 100, 150, 200, 250]) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()

def main():
    (x_train, y_train), (x_test, y_test) = load_mnist()

    #Models
    models = {
        "Sigmoid": {"activation": "sigmoid"},
        "Sigmoid L2": {"activation": "sigmoid", "l2_lambda": 0.01},
        "Sigmoid Dropout": {"activation": "sigmoid", "dropout_rate": 0.5},
        "ReLU": {"activation": "relu"},
        "ReLU L2": {"activation": "relu", "l2_lambda": 0.01},
        "ReLU Dropout": {"activation": "relu", "dropout_rate": 0.5},
        "Sigmoid Early Stopping": {"activation": "sigmoid", "early_stopping": True},
        "ReLU Early Stopping": {"activation": "relu", "early_stopping": True},
    }

    #Train and evaluate
    results = {}
    for name, config in models.items():
        print(f"Training {name}...")
        model_config = {k: v for k, v in config.items() if k != 'early_stopping'}
        model = build_model(**model_config)
        history, batch_losses = train_model(model, x_train, y_train, config.get("early_stopping", False))
        results[name] = {"history": history, "batch_losses": batch_losses}

    #Plot results
    plot_batch_loss(results) 
    plot_validation_error(results) 
    plot_results(results) 

if __name__ == "__main__":
    main()
