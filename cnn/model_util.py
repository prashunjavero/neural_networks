import tensorflow as tf
import numpy as np 

def evaluate(model,test_x,test_y):
    test_loss, test_acc = model.evaluate(test_x,test_y, verbose=2)
    return (test_loss, test_acc)

def get_summary(model):
    model.summarize()

# save the model 
def save_model(model,path):
    #save the model to path 
    model.save(path)

def load_model(path):
    #load the model 
    new_model = tf.keras.models.load_model(path)
    #return the model
    return new_model

def predict(model,img, shape,classes):
    # reshape the image
    img = img.reshape(1,shape[0],shape[1],shape[2])
    # predict the image lavel 
    label = model.predict(img)
    # return the prediction 
    return classes[np.argmax(label)]