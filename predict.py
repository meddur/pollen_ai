import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.enable_eager_execution() #Parce que TF version <2.0 

import numpy as np #funky array stuff
import os #permet os based operations/querries
import glob #What does it do?
from PIL import Image     #Case sensitive  
from skimage.transform import resize #self explanatory package=scikit-image
from skimage.exposure import rescale_intensity #Why is this necessary?
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import itertools # Confusion Matrix
from sklearn.metrics import confusion_matrix # Confusion Matrix
import random #randomize files within directory
import sys



checkpoint_no ="level_1_tricolp_003"
checkpoint_path = "checkpoints/"+checkpoint_no+"/cp-{epoch:04d}.ckpt"
data_path = os.getcwd()+"/images_pre_class"


resolution = 128
choose_random = 20
classes_select = ["bela_502", "bela_505", "bela_508", "bela_511", "bela_514", "bela_517"]
presence = True #could be used to filter if a certain level needs to be classified
max_samples = 600
local_classification = False
ac_function = "softmax"

# =============================================================================
# Definitions
# =============================================================================
#%%
# Definition of process_image
def process_image(filename, width):
    im = Image.open(filename).convert('RGB')                       # open and convert to greyscale
    im = np.asarray(im, dtype=np.float)                          # numpy array
    im_shape = im.shape
    im = resize(im, [width, width], order=1 , mode="constant")    # resize using linear interpolation, replace mode='reflect' by 'constant' otherwise error message 
    im = np.divide(im, 255)                                      # divide to put in range [0 - 1] --- Tensorflow works with values between 0-1
    #im = np.expand_dims(im, axis=2)                             #4dimension: allows BATCH SIZE 1 // I don't think this is bad
    
    im = im[:,:,::-1]
                                                                
    return im, im_shape   

#%%
# Definition of image loading
def load_from_class_dirs(directory, extension, width, norm, min_count=20):
    #norm = TRUE or FALSE /// will rescale intensity between 0-1 (scikit-image)
    #min_count = amount of specimens / classes (Looks like this is a max_count and not a min_count)
    print(" ")
    print("Loading images from the directory './" + directory + "'.")
    print(" ")
    # Init lists
    images = []
    labels = []
    depth_index = []
    filenames = []

    
    # Alphabetically sorted classes
    class_dirs = sorted(glob.glob(directory + "/*"))
    # Load images from each class
    idx = 0 #Set class ID
    for class_dir in class_dirs:

        # Class name
        depth_level = os.path.basename(class_dir)

        
        if (depth_level in classes_select) == presence:  # Import (or not) classes included in classes_select
            num_files = len(os.listdir(class_dir))
            print("%s - %d" % (depth_level, num_files))
            n_samples=0
            

            class_idx = idx
            idx += 1                                    #Increment class ID
            # labels.append(depth_level)                   #Add current class label to labels master list
            
            if local_classification == True and os.path.exists(os.getcwd()+"images_post_class/"+ depth_level) == True:

                overwrite_check = input("Classification folder already exists. Overwrite (y/n)?")
                if overwrite_check != "y": 
                    sys.exit('Error : Checkpoint folder already exists. Aborting')
                else:
                    directory_wipe = glob.glob(os.getcwd()+"images_post_class/"+depth_level+"/*")
                    for f in directory_wipe:         
                        os.remove(f)
            
            if local_classification == True and os.path.exists(os.getcwd()+"images_post_class/"+ depth_level) == False:
                    
                    os.makedirs(os.getcwd()+"images_post_class/"+ depth_level)
            
            # Get the files
            files = sorted(glob.glob(class_dir + "/*." + extension))
            #random.shuffle(files) #shuffle the files within the folder
            random.Random(choose_random).shuffle(files)

            for file in files:
                if n_samples > max_samples: continue
                n_samples +=1
                im, sz = process_image(file, width)
                if norm:
                    im = rescale_intensity(im, in_range='image', out_range=(0.0, 1.0))
                images.append(im)                       #Add current image to images array
                depth_index.append(depth_level)
                

                filenames.append(os.path.basename(os.path.dirname(file)) + os.path.sep + os.path.basename(file))
            print(n_samples)

    # Final clean up
    images = np.asarray(images) #training_images
    depth_index = np.asarray(depth_index)       #training_labels
    num_classes = len(labels)   #Amount of classes
    return images, depth_index, labels, num_classes, filenames

#%%

images, depth_index, labels, num_classes, filenames = load_from_class_dirs(data_path, "png", resolution, False, min_count=20)
#Changed the extension to png, got an error -> im = resize var mode
#I don't think leaving the extension as .tif did anything
images = images[:,:,:,np.newaxis] #4th dimension fix

opt = tf.keras.optimizers.Adam()

#%% 
#########################
#First classification
#########################

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.MaxPooling2D())    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5,seed=7))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))


model.count_params()
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])




latest = tf.train.latest_checkpoint("checkpoints/"+checkpoint_no)
# latest = ("checkpoints/"+checkpoint_no+"/cp-0190.ckpt") #Checkpoint en particulier?
print("LOADING CHECKPOINT " + latest)
pollen_cnn = model.load_weights(latest)

print("got here")

predictions = model.predict(images)


