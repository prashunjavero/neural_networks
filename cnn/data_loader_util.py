import numpy as np
import h5py
import os
from image_utils import load_image
from model_util import predict

def load_dataset_from_h5(train_data_set,test_data_set):
    train_dataset = h5py.File(str(train_data_set), "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(str(test_data_set), "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
 
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def evaluate_dir(root_dir,model,shape,classes, grey_scale=False):
    predictions = []
    predicted_labels = []
    file_names = [] 
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.split('.')[1] == 'jpeg' or f.split('.')[1] == 'png' :
                path = os.path.join(root_dir,f)
                file_names.append(f)
                img = load_image(path,(shape[0],shape[1]),grey_scale)
                label = predict(model,img,shape,classes)
                predictions.append(classes.index(label))
                predicted_labels.append(label)
    return (file_names , predictions , predicted_labels )
