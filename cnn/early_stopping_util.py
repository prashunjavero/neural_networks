import tensorflow as tf
from tensorflow import keras

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        print(logs.get('accuracy'))
        if logs.get('accuracy') > .99:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        # early stop at .999
        print("Reached trainig end returning model ")   