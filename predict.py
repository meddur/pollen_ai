import tensorflow as tf
tf.enable_eager_execution() #Because tensorflow version <2.0 
import numpy as np
import os
import glob
from PIL import Image
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import sys
import shutil
import pandas as pd

##########
# Paths
##########
checkpoint_no ="level_1"
checkpoint_abb_pic = "abb_pic"
checkpoint_acer = "acer"
checkpoint_alnus = 'alnus_shallow'
checkpoint_pinus = "pinus"
checkpoint_tricolp = "tricolp"
checkpoint_tripor = "triporate"
checkpoint_path = "checkpoints_saves/"+checkpoint_no+"/cp-{epoch:04d}.ckpt"


data_path = os.getcwd()+"/fossil_test_sample" #Images to be classified
data_path_post = os.getcwd()+"/images_post_class"   #Where the images will be copied
                                                    #If local_classification = True


directories = [x[0] for x in os.walk(data_path)]
directories = directories[1:]

samples_select = [x[len(data_path)+1:] for x in directories]
samples_select = [int(x) for x in samples_select]
samples_select = sorted(samples_select)



##########
# Parameters & Hyperparameters
##########

batch_size = 32
resolution = 128
choose_random = 20
ac_function = "softmax"
threshold = 0.7 # A prediction value under this threshold means that the
                # image will instead be classified as an unknown image
max_samples = 1200
local_classification = True # If True, locally copies the classified data to its 
                            # own classification folder
local_classification_overwrite = True # This should be left on 
lvl_1_switch = True      # This should be left on 
lvl_2_switch = True      # And so should this
                         # These switches allow the creation of local folders
                         # for each sample and its classes




classes_list = (          # Top level classes
              
                    "abb_pic_mix",
                    "juni_thuya",
                    "npp_mix",
                    "pinus_mix",
                    'tricolp_mix',
                    'tripor_mix',

    )

lbl_dict = {'abb_pic_mix':["abies_b", "picea_mix"],
            'pinus_mix':['pinus_b_mix', 'pinus_s'],
            'tricolp_mix':['acer_mix', 'quercus_r'],
            'tripor_mix':['alnus_mix', 'betula_mix', 'corylus_c', 'eucalyptus'],
            'acer_mix':['acer_r', 'acer_s'],
            'alnus_mix':['alnus_c', 'alnus_r']


    }
  
##########
# Generate some useful lists
##########

lbl_final = []  #Create final label list that includes all sub strata
for label in classes_list:
    
    lbl_final.append(label)
    
    if label in lbl_dict:
        for sub_label in lbl_dict[label]:
            lbl_final.append(sub_label)
            
            if sub_label in lbl_dict:   #add lvl_3 classes
                for sub_sub_label in lbl_dict[sub_label]:
                    lbl_final.append(sub_sub_label) 


lbl_final.append('unknown')
lbl_final = sorted(lbl_final)


# Generate a compilation array to sum pollen counts/sample (exported as csv)
compilation = np.zeros((len(samples_select),len(lbl_final)+1)) 
i = 0
for depth in samples_select:
    compilation[i,0] = depth
    i=i+1



# =============================================================================
# Function definitions
# =============================================================================
#%%

# Definition of process_image
def process_image(filename, width):
    
    """
    This function loads the image and transforms it to a numpy array.
    It resizes it and transforms it [0-1]
    

    Returns
    -------
    im - The rescaled/transformed image data (Numpy)

    """
    
    im = Image.open(filename).convert('RGB')                       # open and convert to greyscale
    im = np.asarray(im, dtype=np.float)                          # numpy array
    im_shape = im.shape
    im = resize(im, [width, width], order=1 , mode="constant")    # resize using linear interpolation, replace mode='reflect' by 'constant' otherwise error message 
    im = np.divide(im, 255)                                      # divide to put in range [0 - 1] --- Tensorflow works with values between 0-1
    im = im[:,:,::-1]
    
    return im, im_shape   
#%%
# Definition of image loading
def load_from_class_dirs(directory, extension, width, norm, min_count=20):
    
    """
    This function loads the image files from the sample directories.
    Every image is labbeled according to its parent directory (sample or level).
    
    Parameters
    ----------
    directory - chosen directory (str)
    extension - file type (str)
    width - passed to process_image function (int)
    norm - Will rescale intensity between 0-1 (scikit-image) (bool)
    min_count - Minimum amount of images/class (int)
    
    Returns
    -------
    images - The image data (numpy array)
    depth_index - The depth levels of each image. Serves as an index (numpy array)
    filenames - The data local filenames (list)

    """
    
    print(" ")
    print("Loading images from the directory '" + directory + "'.")
    print(" ")
    # Init lists
    images = []
    depth_index = []
    filenames = []
    
    # Alphabetically sorted classes    
    class_dirs = sorted(glob.glob(data_path + "/*"))
    class_dirs.sort(key=lambda x: int(''.join(filter(str.isdigit,x))))
    
    # Load images from each class
    for class_dir in class_dirs:

        # Class name
        depth_level = int(os.path.basename(class_dir))
                
        if (depth_level in samples_select) == True:  # Import (or not) classes 
                                                     # included in samples_select
            
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
                

                filenames.append(os.path.basename(os.path.dirname(file)) + 
                                 os.path.sep + os.path.basename(file))
            print(n_samples)


    # Final clean up
    images = np.asarray(images)                 #training_images
    depth_index = np.asarray(depth_index)       #their parent samples/levels
    return images, depth_index, filenames

#%%
def run_local_classification(directory, lbl_list, prediction_data, filenames_list):
    
    """
    Takes the first run of predictions as an input and their corresponding filenames
    If local_classification is set to True, the function will locally copy
    ever file to its corresponding folder (sample/level)
    In either case, this function will tally the compilation np array
    

    Parameters
    ----------
    directory : Sample directory (str)
        
    lbl_list : List of all possible classes (labels) (list)
    
    prediction_data : Prediction array (numpy array)
        
    filenames_list : List of all filenames to move around the local files (list)

    Returns
    -------
    None.

    """
    
    filename_it = 0
    
    # Checks if a classification folder already exists
    # If it already exists, wipe or abort
    # If not, use os.makedirs
    
    #########
    # Generate empty folders (or wipe the existing ones)
    #########
    
    if local_classification_overwrite == True and os.path.exists(data_path_post+"/"+ checkpoint_no) == True:
        overwrite_check = input("Classification folder already exists. Overwrite (y/n)?")
        if overwrite_check != "y": 
            sys.exit('Error : Checkpoint folder already exists. Aborting')
        else:
            shutil.rmtree(data_path_post+"/"+checkpoint_no)
            
    if local_classification == True and os.path.exists(data_path_post+"/"+ checkpoint_no) == False:

            os.makedirs(data_path_post+"/"+ checkpoint_no)
    
    
    for current_depth in samples_select: 
    
        print("Classifying depth "+str(current_depth))
        current_depth_folder = data_path_post+"/"+checkpoint_no+"/"+str(current_depth)
        
        if lvl_1_switch == True:    # Create a sub folder for each depth level
                                    # if this is the first classification run
            
            os.makedirs(current_depth_folder)
            os.makedirs(current_depth_folder+'/unknown')
            
        for lbl in lbl_list:    # Create a sub folder for the local 
                                # classification of each class
            os.makedirs(current_depth_folder+'/'+lbl)
            
        ########
        # Assign images
        ########
        
        for pollen in prediction_data:
            
            
            if pollen[-1] == current_depth: # -1 is the last element in an np array                    

                if pollen[0:len(lbl_list)].max() < threshold: #If the pollen is under threshold
                                                                # During a multiclass classification
    
                    #Local classification & incrementing compilation array
                    if lvl_2_switch == True:    #Safety --- don't classify an already classified pollen as unknown
                        if local_classification == True: shutil.copy(directory+'/'+filenames_list[filename_it], 
                                current_depth_folder+'/unknown')
                        
                        compilation[samples_select.index(current_depth)][lbl_final.index('unknown')+1]+=1
                    
                else:
                    #If the pollen is above threshold during a multiclassification    
                    #Local classification & incrementing compilation array
                    if local_classification == True: shutil.copy(directory+'/'+filenames_list[filename_it], 
                                current_depth_folder+'/'+lbl_list[pollen[0:len(lbl_list)].argmax()])
                    
                    compilation[samples_select.index(current_depth)] \
                    [lbl_final.index(lbl_list[pollen[0:len(lbl_list)].argmax()])+1]+=1    
                
                
                filename_it = filename_it+1
                
            elif pollen[-1] > current_depth: break # Go to next depth 

#%%
def create_masterlist (prediction_data):

    """
    Create a numpy array (masterlist) of every datapoint assigned 
    to a class that contains a substrata and that scaored over the threshold
    The masterlist contains its filename and its assigned parent class

    Parameters
    ----------
    prediction_data : Prediction array (tensors) (numpy array)

    Returns
    -------
    images_sub : Image data of images to be classified in substratas (numpy array)
    filenames_sub : Filenames corresponding to these images (numpy array)
    depth_index_sub : Depths corresponding to these images (numpy array)
    masterlist_sub : Masterlist. Contains filename and parent class (numpy array)
    
    """
    
    copy_it = 0
    copy_index = []
    pollen_index = []
    filenames_np = np.asarray(filenames)

    
    for pollen in prediction_data:
        if classes_list[pollen[0:len(classes_list)].argmax()] in lbl_dict \
        and pollen[0:len(classes_list)].max() > threshold:
            
            copy_index.append(copy_it)
            pollen_index.append(str(classes_list[pollen[0:len(classes_list)].argmax()]))
            
        copy_it = copy_it+1
    
    
    
    images_sub = np.copy(images[copy_index,])
    depth_index_sub = np.copy(depth_index[copy_index,])
    
    filenames_sub = np.copy(filenames_np[copy_index,])
    filenames_sub = filenames_sub[:,np.newaxis]
    
    
    pollen_index = np.asarray(pollen_index)
    pollen_index = pollen_index[:,np.newaxis]
    
    masterlist_sub = np.concatenate((filenames_sub, pollen_index),axis = 1)
    
    return images_sub, filenames_sub, depth_index_sub, masterlist_sub

#%%
def get_scaled_prediction (logits, temperature):

    """
    Calibrate the predictions w/ temperature scaling
    
    Parameters
    ----------
    logits : Prediction array pre-calibration. (numpy array)
    temperature : Temperature value to calibrate (>0) (int)
    
    Returns
    -------
    scld_per : Calibrated predictions (scaled to [0-1]) (numpy array)
    
    """
            
    logits_w_temp = tf.divide(logits, temperature)

    scld_predict = np.exp(logits_w_temp) / np.sum(np.exp(logits_w_temp),    #Scale the predictions
                                                         axis=-1, keepdims=True)
    
    scld_predict = np.copy(logits)
    
    scld_per = np.where(np.max(scld_predict, axis=0)==0, scld_predict,
                        scld_predict*1./np.max(scld_predict, axis=0))
        
    return scld_per 


#%%
def sub_level_prediction (masterlist, images, branch, checkpoint_sub, depth_index, use_latest_checkpoint, temperature):
    
    """
    Generate a subselection of the main image data fed from an upper level branch.
    Loads the new model's weights.
    Predicts the data subselection into its substrata (sub classes).
    
    Parameters
    ----------
    masterlist : Prediction data from an upper level (numpy array)
    images : The corresponding image data (numpy array)
    branch : The parent branch name (str)
    checkpoint_sub : Checkpoint name (local) (str)
    depth_index : Corresponding depth data (numpy array)
    use_latest_checkpoint : If True, will load latest checkpoint in the folder
                            Can be an integer (Particular checkpoint number
                                               rather than the use the latest)
    temperature : Temperature value to calibrate the data (>0) (int)
                    Passed to get_scaled_prediction
    Returns
    -------
    predictions_sub : The corresponding predictions, calibrated. (numpy array)
    """
    
    index_extract = np.where(masterlist == branch)[0] #[filename][prediction]
    images_extract = np.copy(images_sub[index_extract,])
    filenames_extract = np.copy(filenames_sub[index_extract,])
    filenames_extract = filenames_extract.flatten()
    depth_index_extract = np.copy(depth_index[index_extract,])
    
    #Load model && predict
    

    model = tf.keras.models.load_model(filepath = ("checkpoints_saves/"+checkpoint_sub+"/"+checkpoint_sub+"_model"))

    predictions_sub_float = model.predict(images_extract)
    
    print("branch ", branch, "has ", len(predictions_sub_float), " items")
    
    if len(index_extract) > 0:
        
        # if ac_function == 'softmax':
        scld_per = get_scaled_prediction(logits = predictions_sub_float, temperature = temperature)
        predictions_sub = np.append(scld_per, depth_index_extract, axis = -1) 
        # Adds depth data to predictions along the last axis
        
        # else: 
        #     predictions_sub = np.append(predictions_sub_float, depth_index_extract, axis = -1)

        run_local_classification(data_path, lbl_dict[branch], predictions_sub, filenames_extract)
        

    else: predictions_sub = predictions_sub_float
    return predictions_sub

#%%
def update_masterlist (masterlist, prediction_data, branch):
    """
    Updates the masterlist with the new prediction data

    Parameters
    ----------
    masterlist : List of filenames and their corresponding class (numpy array)
    prediction_data : New prediction data used to update the masterlist (numpy array)
    branch : Parent branch name (str)

    Returns
    -------
    masterlist : Updated masterlist with the new class names (numpy array)

    """
    i = 0

    for pollen in masterlist:
        
        if pollen[1] == branch:

            if np.max(prediction_data[i,:len(lbl_dict[branch])]) > threshold: 
                # If the pollen prediction certitude is above threshold, update the masterlist
                pollen[1] = lbl_dict[branch][int(np.argmax(prediction_data[i,:len(lbl_dict[branch])]))]
                    
            i = i+1

    return masterlist     

#%%

def load_transfer_learning():
    """
    Load the transfer learning model layers (VGG16)
    
    Returns
    -------
    base_model : The transfer learning model layers (tf.layer object)

    """
    base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

    #Lock layers
    for layer in base_model.layers[0:3]:
        layer.trainable = False

    base_model.trainable = False # For good measure
    
    return base_model    
#%%
                         
def generate_model(depth, output):
    
    """
    Generates either the shallow or deep model.

    Parameters
    ----------
    depth : Chosen depth. Either "shallow" or "deep" (str)
        
    output : The amount of output classes needed for the model (int)

    Returns
    -------
    model : The Tensorflow model (tf.model object)
        

    """
    base_model = load_transfer_learning()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(resolution,resolution,3))) #Add this layer when importing a model
    model.add(base_model.layers[3])
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D())
    
    
    if depth == "shallow":
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5,seed=7))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(output, activation = ac_function))
    
    if depth == "deep":
        
        model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D())    
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5,seed=7))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(output, activation=ac_function))
            
    return model

#%%

def reset_inator(model):
    """
    Resets the session and deletes the previous model.
    This allows the script to load another model and its weights.
    This is necessary because this runs with tensorflow v.<2

    Parameters
    ----------
    model : The tensorflow model (tf.model object)

    """
    del model
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
#%%

#########################
# First classification
#########################

# Load images to classify
images, depth_index, filenames = load_from_class_dirs(data_path, "png", resolution, False, min_count=20)

# Choose optimizer
opt = tf.keras.optimizers.Adam()



#%%


# os.environ['KMP_DUPLICATE_LIB_OK']='True' #MacOS (Intel) only

model = generate_model(depth = "deep", output=len(classes_list))

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Load the model save


model = tf.keras.models.load_model(os.path.dirname(checkpoint_path)+"/"
                                           +checkpoint_no+"_model")





#%%
############
# Generate first predictions
############

predictions_float = model.predict(images)
scaled_predictions = get_scaled_prediction(predictions_float, temperature = 0.195)


depth_index = depth_index[:,np.newaxis] #reshapes depth index -> allows np.append
predictions = np.append(scaled_predictions, depth_index,1) #adds depth data to predictions



#%%

# Localy classify + increment the output csv
run_local_classification(data_path, classes_list, predictions, filenames)  

local_classification_overwrite = False #So that when we loop back to classify
                    # deeper branches, we don't trigger the overwrite safety
lvl_1_switch = False



#%%

# Create a masterlist (filenames + predicted class)
images_sub, filenames_sub, depth_index_sub, masterlist_sub = create_masterlist(predictions)



#%%
####################################
#CLASSIFY ABIES & PICEA
####################################

print("Classifying Abies & Picea")

reset_inator(model)

model = generate_model(depth = "deep", output=len(lbl_dict['abb_pic_mix']))
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])




predictions_abb_pic = sub_level_prediction(masterlist_sub, images_sub, "abb_pic_mix", 
                                            checkpoint_abb_pic, depth_index_sub, use_latest_checkpoint = True,
                                            temperature = 0.345)


masterlist_sub = update_masterlist(masterlist_sub, predictions_abb_pic, 'abb_pic_mix')

#%%
####################################
#CLASSIFY PINUS
####################################

print("Classifying Pinus")

reset_inator(model)

model = generate_model(depth = "deep", output=len(lbl_dict['pinus_mix']))
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

predictions_pinus = sub_level_prediction(masterlist_sub, images_sub, "pinus_mix", 
                                         checkpoint_pinus, depth_index_sub, use_latest_checkpoint = True,
                                         temperature = 0.348)

masterlist_sub = update_masterlist(masterlist_sub, predictions_pinus, 'pinus_mix')

#%%
####################################
#CLASSIFY TRICOLPORATE
####################################

print("Classifying Tricolporate")

reset_inator(model)

model = generate_model(depth = "deep", output=len(lbl_dict['tricolp_mix']))
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

predictions_tricolp = sub_level_prediction(masterlist_sub, images_sub, "tricolp_mix", 
                                           checkpoint_tricolp, depth_index_sub, use_latest_checkpoint = 550,
                                           temperature = 0.486)

masterlist_sub = update_masterlist(masterlist_sub, predictions_tricolp, 'tricolp_mix')


#%%
####################################
#CLASSIFY TRIPORATE
####################################

print("Classifying Triporate")

reset_inator(model)

model = generate_model(depth = "deep", output=len(lbl_dict['tripor_mix']))
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

predictions_tripor = sub_level_prediction(masterlist_sub, images_sub, "tripor_mix", 
                                          checkpoint_tripor, depth_index_sub, use_latest_checkpoint = True,
                                          temperature = 0.278)

masterlist_sub = update_masterlist(masterlist_sub, predictions_tripor, 'tripor_mix')

#%%
####################################
#CLASSIFY ACER
####################################
lvl_2_switch = False #From now on, Pollens won't be classified as unknown nor will they be locally copied

print("Classifying Acer")

reset_inator(model)

model = generate_model(depth = "deep", output=len(lbl_dict['acer_mix']))
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

predictions_acer = sub_level_prediction(masterlist_sub, images_sub, "acer_mix", 
                                        checkpoint_acer, depth_index_sub, use_latest_checkpoint = True,
                                        temperature = 0.403)

masterlist_acer = np.copy(masterlist_sub)
masterlist_sub = update_masterlist(masterlist_sub, predictions_acer, 'acer_mix')


#%%
####################################
#CLASSIFY ALNUS
####################################

print("Classifying Alnus")

reset_inator(model)

model = generate_model(depth = "shallow", output=len(lbl_dict['alnus_mix']))
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


predictions_alnus = sub_level_prediction(masterlist_sub, images_sub, "alnus_mix", 
                                         checkpoint_alnus, depth_index_sub, use_latest_checkpoint = 170,
                                         temperature = 0.309)

masterlist_sub = update_masterlist(masterlist_sub, predictions_alnus, 'alnus_mix')


#%%
###################
# Generate output datafile (csv)
###################

header_df = lbl_final
header_df.insert(0,'depth')


pd.DataFrame(compilation).to_csv(data_path_post + "/"+str(min(samples_select))+"_"+str(max(samples_select))+"_taxa_sum.csv", index = None, header=header_df)

