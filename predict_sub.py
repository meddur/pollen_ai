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
import pandas as pd


checkpoint_no ="transfer_1200_npp_trip_deep"
checkpoint_abb_pic = "transfer_abb_pic_sigmoid"
checkpoint_pinus = "transfer_pinus_sigmoid"
checkpoint_tricolp = "transfer_tricolp_sigmoid"
checkpoint_tripor = "transfer_tripor_small"
checkpoint_path = "checkpoints/"+checkpoint_no+"/cp-{epoch:04d}.ckpt"
data_path = os.getcwd()+"/images_pre_honors"
data_path_post = os.getcwd()+"/images_post_class"



batch_size = 32
resolution = 128
choose_random = 20


directories = [x[0] for x in os.walk(data_path)]
directories = directories[1:]
classes_select = [x[len(data_path)+1:] for x in directories]
classes_select = [int(x) for x in classes_select]
classes_select = sorted(classes_select)





threshold = 0.5
presence = True #could be used to filter if a certain level needs to be classified
max_samples = 60000
local_classification = False #If True, localy copies the classified data to its own classification folder

local_classification_overwrite = True #This should be left on
lvl_0_switch = True      #This should be left on 



ac_function = "softmax"
lbl_pretty = (
                    "abb_pic_mix",
                    # "abies_b",
                    # "acer_mix",
                    # "acer_r",
                    # "acer_s",
                    # 'alnus_mix',
                    # "alnus_c",
                    # "alnux_r",
                    # "betula_mix",
                    # "corylus_c",
                    # "eucalyptus",
                    "juni_thuya",
                    "npp_mix",
                    # "picea_mix",
                    # "pinus_b_mix",
                    "pinus_mix",
                    # "pinus_s",
                    # "populus_d",
                    # "quercus_r",
                    'tricolp_mix',
                    'tripor_mix',
                    
                    # 'tsuga',
                    #'vesiculate_mix',
    )

lbl_dict = {'abb_pic_mix':["abies_b", "picea_mix"],
            'pinus_mix':['pinus_b_mix', 'pinus_s'],
            'tricolp_mix':['acer_mix', 'quercus_r'],
            'tripor_mix':['alnus_mix', 'betula_mix', 'corylus_c', 'eucalyptus'],

    }
  

#CREATE LIST OF ALL LABELS
lbl_final = []  #Create final label list that includes all sub strata
for label in lbl_pretty:
    
    lbl_final.append(label)
    
    if label in lbl_dict:
        for sub_label in lbl_dict[label]:
            lbl_final.append(sub_label)
    
    # if label not in lbl_dict:
    #     lbl_final.append(label)

lbl_final.append('unknown')
lbl_final = sorted(lbl_final)


#CREATE COMPILATION ARRAY TO SUM POLLEN COUNTS
compilation = np.zeros((len(classes_select),len(lbl_final)+1)) #Create final compilation numpy matrix

#row then column
i = 0
for depth in classes_select:
    compilation[i,0] = depth
    i=i+1



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
    print("Loading images from the directory '" + directory + "'.")
    print(" ")
    # Init lists
    images = []
    depth_index = []
    filenames = []


    
    # Alphabetically sorted classes
    
    class_dirs = sorted(glob.glob(data_path + "/*")) # STRINGS DO NOT SORT THE SAME AS INTEGERS
    class_dirs.sort(key=lambda x: int(''.join(filter(str.isdigit,x)))) # SORT LIKE A HUMAN 
    print("DON'T FORGET TO ADD A SAFETY SO THAT POLLENS DONT GET CLASSIFIED IN THE WRONG FOLDER")
    
    
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
# Definition of image loading
def load_from_sub(directory, extension, width, norm, min_count=20):
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
    class_dirs.sort(key=lambda x: int(''.join(filter(str.isdigit,x)))) # SORT LIKE A HUMAN 
    
    
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
def run_local_classification(directory, lbl_list, prediction_data, filenames_list):
    
    filename_it = 0
    
    #Checks if a classification folder already exists
    #If it already exists, wipe or abort
    if local_classification_overwrite == True and os.path.exists(data_path_post+"/"+ checkpoint_no) == True:
        overwrite_check = input("Classification folder already exists. Overwrite (y/n)?")
        if overwrite_check != "y": 
            sys.exit('Error : Checkpoint folder already exists. Aborting')
        else:
            shutil.rmtree(data_path_post+"/"+checkpoint_no)
            
    if local_classification == True and os.path.exists(data_path_post+"/"+ checkpoint_no) == False:

            os.makedirs(data_path_post+"/"+ checkpoint_no)
    
    
    for current_depth in classes_select: 
    
        print("Classifying depth "+str(current_depth))
        current_depth_folder = data_path_post+"/"+checkpoint_no+"/"+str(current_depth)
        
        if lvl_0_switch == True:    #create a sub folder for each depth level if this is the first classification round
            
            os.makedirs(current_depth_folder)
            os.makedirs(current_depth_folder+'/unknown')
            
        for lbl in lbl_list:
            os.makedirs(current_depth_folder+'/'+lbl)
        
        for pollen in prediction_data:
            
            # print(str(pollen[len(lbl_list)]) + " is equal to " + str(current_depth))
            
            if pollen[-1] == current_depth: # -1 is the last element in an np array
                
                if ac_function == 'sigmoid': #CLEAN THIS
                    
                    #Local classification & incrementing compilation array
                    if local_classification == True: shutil.copy(directory+'/'+filenames_list[filename_it],
                                current_depth_folder+'/'+lbl_list[pollen[0]])
                    
                    compilation[classes_select.index(current_depth)] \
                    [lbl_final.index(lbl_list[pollen[0]])+1]+=1   
                    
                
                elif pollen[0:len(lbl_list)].max() <= threshold:
                    
                    #Local classification & incrementing compilation array
                    if local_classification == True: shutil.copy(directory+'/'+filenames_list[filename_it], 
                            current_depth_folder+'/unknown')
                    
                    compilation[classes_select.index(current_depth)][lbl_final.index('unknown')+1]+=1
                    
                    

                else:
                    
                    #Local classification & incrementing compilation array
                    if local_classification == True: shutil.copy(directory+'/'+filenames_list[filename_it], 
                                current_depth_folder+'/'+lbl_list[pollen[0:len(lbl_list)].argmax()])
                    
                    compilation[classes_select.index(current_depth)] \
                    [lbl_final.index(lbl_list[pollen[0:len(lbl_list)].argmax()])+1]+=1    
                
                
                filename_it = filename_it+1
                
            elif pollen[-1] > current_depth: break



#%%
def create_masterlist (checkpoint_level):
    
    # latest = tf.train.latest_checkpoint("checkpoints/"+checkpoint_level)
    # latest = ("checkpoints/"+checkpoint_no+"/cp-0300.ckpt") #Checkpoint en particulier?
    
    # second_cnn = model.load_weights(latest)
    
    copy_it = 0
    copy_index = []
    pollen_index = []
    filenames_np = np.asarray(filenames)
    # masterlist_lvl2 = np.asarray()
    
    
    for pollen in predictions:
        if lbl_pretty[pollen[0:len(lbl_pretty)].argmax()] in lbl_dict and pollen[0:len(lbl_pretty)].max() > threshold:
            # WE DONT WANT TO CLASSIFY UNKNOWN POLLENS ON THE SECOND CLASSIFICATION RUN
            copy_index.append(copy_it)
            pollen_index.append(str(lbl_pretty[pollen[0:len(lbl_pretty)].argmax()]))
            
        copy_it = copy_it+1
    
    
    
    images_lvl2 = np.copy(images[copy_index,])
    depth_index_lvl2 = np.copy(depth_index[copy_index,])
    
    filenames_lvl2 = np.copy(filenames_np[copy_index,])
    filenames_lvl2 = filenames_lvl2[:,np.newaxis]
    
    
    pollen_index = np.asarray(pollen_index)
    pollen_index = pollen_index[:,np.newaxis]
    
    masterlist_lvl2 = np.concatenate((filenames_lvl2, pollen_index),axis = 1)
    
    return copy_index, images_lvl2, filenames_lvl2, pollen_index, depth_index_lvl2, masterlist_lvl2

#%%
def sub_level_prediction (masterlist, images, branch, checkpoint_sub, depth_index):
    
    #Create sub_image list
    
    index_extract = np.where(masterlist == branch)[0] #Change this
    images_extract = np.copy(images_lvl2[index_extract,])
    filenames_extract = np.copy(filenames_lvl2[index_extract,])
    filenames_extract = filenames_extract.flatten()
    depth_index_extract = np.copy(depth_index[index_extract,])
    
    #Load checkpoint && predict
    latest = tf.train.latest_checkpoint("checkpoints/"+checkpoint_sub)
    # latest = ("checkpoints/"+checkpoint_sub+"/cp-0300.ckpt") #Checkpoint en particulier?
    print("LOADING CHECKPOINT " + latest)
    
    pollen_cnn = model.load_weights(latest)
    
    if ac_function == 'sigmoid':    
        predictions_lvl2_float = (model.predict(images_extract) >= 0.5).astype("int64")
        predictions_lvl2_float = predictions_lvl2_float[0:,0]
        predictions_lvl2_float = predictions_lvl2_float[:,np.newaxis]
        
    else: predictions_lvl2_float = model.predict(images_extract)
    
    
    
    predictions_lvl2 = np.append(predictions_lvl2_float, depth_index_extract, 1) #adds depth data to predictions along the last axis
    
    
    run_local_classification(data_path, lbl_dict[branch], predictions_lvl2, filenames_extract)

    return predictions_lvl2


#%%

images, depth_index, filenames = load_from_class_dirs(data_path, "png", resolution, False, min_count=20)

# images = images[:,:,:,np.newaxis] #4th dimension fix

opt = tf.keras.optimizers.Adam()

# base_model = VGG16(input_shape = (resolution, resolution, 3), weights = "imagenet", include_top=False)  
base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

#Lock layers
for layer in base_model.layers[0:3]:
    layer.trainable = False

base_model.trainable = False

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
model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5,seed=7))
# model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(len(lbl_pretty), activation=ac_function))

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# model.count_params()

# model.summary()

latest = tf.train.latest_checkpoint("checkpoints/"+checkpoint_no)
# latest = ("checkpoints/"+checkpoint_no+"/cp-0300.ckpt") #Checkpoint en particulier?
print("LOADING CHECKPOINT " + latest)
pollen_cnn = model.load_weights(latest)
predictions_float = model.predict(images)
depth_index = depth_index[:,np.newaxis] #reshapes depth index -> allows np.append
predictions = np.append(predictions_float, depth_index,1) #adds depth data to predictions




#%%
# filenames = np.asarray(filenames)
# filenames = filenames[:,np.newaxis]
# predictions_filenames = np.append(predictions_float, filenames, 1)

# if local_classification == True:
#     run_local_classification(data_path, lbl_pretty, predictions, filenames)  
#     local_classification_overwrite = False #So that when we loop back to classify deeper branches, we don't trigger the overwrite safety
#     lvl_0_switch = False


run_local_classification(data_path, lbl_pretty, predictions, filenames)  
local_classification_overwrite = False #So that when we loop back to classify deeper branches, we don't trigger the overwrite safety
lvl_0_switch = False



#%%
copy_index, images_lvl2, filenames_lvl2, pollen_index, depth_index_lvl2, masterlist_lvl2 = create_masterlist(checkpoint_no) #run second prediction from lvl 2 pollens

#%%
####################################
#CLASSIFY ABIES & PICEA
####################################

print("Classifying Abies & Picea")

del model# THERE COULD BE OTHER THINGS TO DELETE/ RESET THE SEED / RESET DEFAULT GRAPH
del base_model

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
#reset_seeds()


base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

#Lock layers
for layer in base_model.layers[0:3]:
    layer.trainable = False

base_model.trainable = False

ac_function = 'sigmoid'

model = tf.keras.models.Sequential()

model.add(base_model.layers[3])
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.MaxPooling2D())    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5,seed=7))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(len(lbl_dict['abb_pic_mix']), activation = ac_function))

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

predictions_abb_pic = sub_level_prediction(masterlist_lvl2, images_lvl2, "abb_pic_mix", checkpoint_abb_pic, depth_index_lvl2)
#%%

#%%
####################################
#CLASSIFY PINUS
####################################

print("Classifying Pinus")

del model# THERE COULD BE OTHER THINGS TO DELETE/ RESET THE SEED / RESET DEFAULT GRAPH
del base_model

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
#reset_seeds()


base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

#Lock layers
for layer in base_model.layers[0:3]:
    layer.trainable = False

base_model.trainable = False

ac_function = 'sigmoid'

model = tf.keras.models.Sequential()

model.add(base_model.layers[3])
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.MaxPooling2D())    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5,seed=7))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(len(lbl_dict['pinus_mix']), activation = ac_function))

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

predictions_pinus = sub_level_prediction(masterlist_lvl2, images_lvl2, "pinus_mix", checkpoint_pinus, depth_index_lvl2)


#%%
####################################
#CLASSIFY TRICOLPORATE
####################################

print("Classifying Triporate")

del model# THERE COULD BE OTHER THINGS TO DELETE/ RESET THE SEED / RESET DEFAULT GRAPH
del base_model

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
#reset_seeds()


base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

#Lock layers
for layer in base_model.layers[0:3]:
    layer.trainable = False

base_model.trainable = False

ac_function = 'sigmoid'

model = tf.keras.models.Sequential()

model.add(base_model.layers[3])
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.MaxPooling2D())    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5,seed=7))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(len(lbl_dict['tricolp_mix']), activation = ac_function))

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

predictions_tricolp = sub_level_prediction(masterlist_lvl2, images_lvl2, "tricolp_mix", checkpoint_tricolp, depth_index_lvl2)

#%%
####################################
#CLASSIFY TRIPORATE
####################################

print("Classifying Triporate")

del model# THERE COULD BE OTHER THINGS TO DELETE/ RESET THE SEED / RESET DEFAULT GRAPH
del base_model

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
#reset_seeds()


base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

#Lock layers
for layer in base_model.layers[0:3]:
    layer.trainable = False

base_model.trainable = False

ac_function = 'softmax'

model = tf.keras.models.Sequential()

model.add(base_model.layers[3])
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for 128x128
model.add(tf.keras.layers.MaxPooling2D())    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5,seed=7))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(len(lbl_dict['tripor_mix']), activation = ac_function))

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

predictions_tripor = sub_level_prediction(masterlist_lvl2, images_lvl2, "tripor_mix", checkpoint_tripor, depth_index_lvl2)







#%%

header_df = lbl_final
header_df.insert(0,'depth')


pd.DataFrame(compilation).to_csv(data_path_post +"/"+ checkpoint_no + "/taxa_sum.csv", index = None, header=header_df)
