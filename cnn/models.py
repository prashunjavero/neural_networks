import tensorflow as tf

def binary_classifier(optimizer='adam',loss='binary_crossentropy'):
    # creat the model 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),strides=(1, 1), padding='valid',input_shape=(64,64,3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32,(3,3),strides=(1, 1), padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128 , activation = tf.nn.relu ),
        tf.keras.layers.Dense(1,  activation = tf.nn.sigmoid)
    ])
     # add the loss function and the optimizer for gradient descent 
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    return  model


def fashion_mnist_classifier(optimizer='adam',loss='binary_crossentropy'):
    # creat the model 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128 , activation = tf.nn.relu ),
        tf.keras.layers.Dense(10,  activation = tf.nn.softmax)
    ])

    # add the loss function and the optimizer for gradient descent 
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
                      
    return  model