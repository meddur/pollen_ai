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

import shutil

checkpoint_no ="TRANSFER_test_lyr3_augs2"
checkpoint_ves = ""
checkpoint_tri = ""
checkpoint_alnus = ""
checkpoint_path = "checkpoints/"+checkpoint_no+"/cp-{epoch:04d}.ckpt"
data_path = os.getcwd()+"/images_pre_class"
data_path_post = os.getcwd()+"/images_post_class"


batch_size = 32
resolution = 128
choose_random = 20
classes_select = [502, 505, 508, 511, 514, 517]
threshold = 0.80
presence = True #could be used to filter if a certain level needs to be classified
max_samples = 60000
local_classification = True
ac_function = "softmax"
lbl_pretty = (
                    "abb_pic_mix",
                    # "abies_b",
                    # "acer_mix",
                    # "acer_r",
                    # "acer_s",
                    'alnus_mix',
                    # "alnus_c",
                    # "alnux_r",
                    "betula_mix",
                    "corylus_c",
                    "eucalyptus",
                    "juni_thuya",
                    # "picea_mix",
                    # "pinus_b_mix",
                    "pinus_mix",
                    # "pinus_s",
                    # "populus_d",
                    # "quercus_r",
                    'tricolp_mix',
                    #'vesiculate_mix',
    )



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
    print("Loading images from the directory '." + directory + "'.")
    print(" ")
    # Init lists
    images = []
    depth_index = []
    filenames = []

    
    # Alphabetically sorted classes
    class_dirs = sorted(glob.glob(directory + "/*"))
    # Load images from each class
    for class_dir in class_dirs:

        # Class name
        
        
        depth_level = int(os.path.basename(class_dir))
                
        
        if (depth_level in classes_select) == presence:  # Import (or not) classes included in classes_select
            num_files = len(os.listdir(class_dir))
            print("%s - %d" % (depth_level, num_files))
            n_samples=0
            


            # Get the files
            files = sorted(glob.glob(class_dir + "/*." + extension))

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
    return images, depth_index, filenames

#%%
# Definition of classification
def run_local_classification(directory):
    
    filename_it = 0
    
    #Checks if a classification folder already exists
    #If it already exists, wipe or abort
    if local_classification == True and os.path.exists(os.getcwd()+"/images_post_class/"+ checkpoint_no) == True:
        
        overwrite_check = input("Classification folder already exists. Overwrite (y/n)?")
        if overwrite_check != "y": 
            sys.exit('Error : Checkpoint folder already exists. Aborting')
        else:
            shutil.rmtree(os.getcwd()+"/images_post_class/"+checkpoint_no)
            
    if local_classification == True and os.path.exists(os.getcwd()+"/images_post_class/"+ checkpoint_no) == False:

            os.makedirs(os.getcwd()+"/images_post_class/"+ checkpoint_no)
    
    
    for current_depth in classes_select: #create a sub folder for each depth level
    
        print("Classifying depth "+str(current_depth))
        current_depth_folder = os.getcwd()+'/images_post_class/'+checkpoint_no+"/"+str(current_depth)
        os.makedirs(current_depth_folder)
        os.makedirs(current_depth_folder+'/unknown')
        for lbl in lbl_pretty:
            os.makedirs(current_depth_folder+'/'+lbl)
        
        for pollen in predictions:
            # while int(str(pollen[8])[4:])<=
            if pollen[8] == current_depth:
                
                if pollen[0:8].max() <= threshold:
                    shutil.copy(directory+'/'+filenames[filename_it], 
                            current_depth_folder+'/unknown')
                    
                else:
                    shutil.copy(directory+'/'+filenames[filename_it], 
                                current_depth_folder+'/'+lbl_pretty[pollen[0:8].argmax()])
                
                filename_it = filename_it+1
                    
                
                # print(pollen[0:8].argmax())

#%%

images, depth_index, filenames = load_from_class_dirs(data_path, "png", resolution, False, min_count=20)
#Changed the extension to png, got an error -> im = resize var mode
#I don't think leaving the extension as .tif did anything
# images = images[:,:,:,np.newaxis] #4th dimension fix

opt = tf.keras.optimizers.Adam()

# base_model = VGG16(input_shape = (resolution, resolution, 3), weights = "imagenet", include_top=False)  
base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

#Lock layers
for layer in base_model.layers[0:3]:
    layer.trainable = False

#%% 
#########################
#First classification
#########################

model = tf.keras.models.Sequential()
model.add(base_model.layers[3])
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))
# model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.MaxPooling2D())    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5,seed=7))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(len(lbl_pretty), activation=ac_function))


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
# model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
# model.add(tf.keras.layers.MaxPooling2D())    
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dropout(0.5,seed=7))
# model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.Dense(len(lbl_pretty), activation=ac_function))

# model.count_params()

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# model.summary()



latest = tf.train.latest_checkpoint("checkpoints/"+checkpoint_no)
# latest = ("checkpoints/"+checkpoint_no+"/cp-0190.ckpt") #Checkpoint en particulier?
print("LOADING CHECKPOINT " + latest)
pollen_cnn = model.load_weights(latest)


predictions_float = model.predict(images)
depth_index = depth_index[:,np.newaxis] #reshapes depth index -> allows np.append
predictions = np.append(predictions_float, depth_index,1) #adds depth data to predictions


if local_classification == True:
    run_local_classification(data_path)  




# filenames_debug = np.asarray(filenames)
# filenames_debug = filenames_debug[:,np.newaxis]
# predictions_debug = np.append(predictions, filenames_debug, 1)

# type(predictions_debug[4,0])
# predictions_debug[4,0:8]

# tf.argmax(predictions_debug, axis=[4,0:8])
# predictions_debug[4,0:8].argmax()
# max(predictions_debug[4,0:8])

# predictions[4,0:8].max()
