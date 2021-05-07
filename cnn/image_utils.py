
import matplotlib.pyplot as plt
import  numpy as np
import cv2 

def display_image(images,labels,classes,index):
    plt.imshow(images[index])
    print ("y = " + str(labels[0, index]) + ". It's a " + classes[labels[0, index]].decode("utf-8") + " picture.")

def load_image(filename,size,grey_scale=False):
	# load the image
    img = cv2.imread(filename)
    # reshape into a single sample with 1 channel
    if(type(img) == type(None)):
        pass
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    if grey_scale:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert to array
    img = np.array(img)
    # prepare pixel data
    img = img.astype('float32')
    # normalize the data 
    img = img / 255.0
    # return the image 
    return img
