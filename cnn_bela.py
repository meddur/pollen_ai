# =============================================================================
# CNN Bela
# =============================================================================

import tensorflow as tf
tf.enable_eager_execution() #Because TF version <2.0 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
from PIL import Image
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import random
import sys









print("TensorFlow version " + tf.__version__)

# =============================================================================
# Settings
# =============================================================================

resolution = 128             
presence = True                # True = classes included in classes_select WILL be included in the model - False = WILL NOT be included
classes_select = [
                    # "abb_pic_mix",
                    # "abies_b",
                    # "acer_mix",
                    # "acer_r",
                    # "acer_s",
                    'alnus_mix',
                    # "alnus_c",
                    # "alnus_r",
                        "betula_mix",
                        "corylus_c",
                        "eucalyptus",
                        # "juni_thuya",
                    # "npp_mix",
                    # "picea_mix",
                    # "pinus_b_mix",
                    # "pinus_b_sans_resinosa",
                    # "pinus_mix",
                    # "pinus_s",
                    # "populus_d",
                    # "quercus_r",
                    # 'tricolp_mix',
                    # 'tripor_mix',
                    # 'tsuga',
                    #'vesiculate_mix',

                    
        ]                        # Classes you want to train the network on


#Name of the directory
# Your dataset here
path_folder_image = os.getcwd()+"/pollen_dataset/your_dataset_here"




#Parameters

max_samples = 110
n_epochs = 600
choose_random = 20
ac_function = "softmax"
batch_size = 32

use_deep_model = True

#Checkpoint setup

will_train = True       # False = load an already existing model
will_save = True
load_weights = False # True: load weights directly instead of loading the model object
threshold = 0.7

checkpoint_no ="triporate"
checkpoint_path = "checkpoints_saves/"+checkpoint_no+"/cp-{epoch:04d}.ckpt"


latest = tf.train.latest_checkpoint("checkpoints_saves/"+checkpoint_no)
# latest = ("checkpoints_saves/"+checkpoint_no+"/cp-0550.ckpt") # Use to load a specific checkpoint instead of the default

#########################################################################



if will_save == True and os.path.exists(os.getcwd()+"/checkpoints_saves/"+
                                        checkpoint_no) == True:

    overwrite_check = input("Checkpoint already exists. Overwrite (y/n)?")
    if overwrite_check != "y": 
        sys.exit('Error : Checkpoint folder already exists. Aborting')
    else:
        directory_wipe = glob.glob(os.getcwd()+"checkpoints_saves/"+
                                   checkpoint_no+"/*")
        for f in directory_wipe:         
            os.remove(f)
    

notes=""



notes_ckpt = "Checkpoint : "+checkpoint_no+"\nResolution : " + str(resolution) + \
    "\nClasses : " + str(classes_select) + "\nMax samples : " + str(max_samples) + \
        "\nEpochs : " + str(n_epochs) + "\nActivation function : " + ac_function + \
                "\nUse deep model : " + str(use_deep_model) + \
            "\nSeed : " + str(choose_random) + "\nNotes : "+ notes + "\n Path folder image : "+path_folder_image

print(notes_ckpt)



if ac_function == "softmax" and len(classes_select) == 2: print("SOFTMAX FUNCTION USED FOR BINARY CLASSIFICATION")
elif ac_function == "sigmoid" and len(classes_select) > 2: print("SIGMOID FUNCTION USED FOR NON-BINARY CLASSIFICATION")


# =============================================================================
# Definitions
# =============================================================================
# %%

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
    im = resize(im, [width, width], order=1 , mode="constant")    # resize using linear interpolation, replace mode='reflect' by 'constant' otherwise error message 
    im = np.divide(im, 255)                                      # divide to put in range [0 - 1] --- Tensorflow works with values between 0-1
    #im = np.expand_dims(im, axis=2)                             #4dimension: allows BATCH SIZE 1 // I don't think this is bad
    
    im = im[:,:,::-1]
                                                                
    return im  
#%%

def load_from_class_dirs(directory, extension, width, norm):
    
    """
    This function loads the image files from the training directory.
    Every image is labbeled according to its parent directory.

    Returns
    -------
    images - The image data (numpy array)
    cls - The label indexes (numpy array)
    labels - The label strings referenced by cls (list)
    filenames - The data local filenames (list)

    """
    #norm = TRUE or FALSE /// will rescale intensity between 0-1 (scikit-image)
    #min_count = amount of specimens / classes (Looks like this is a max_count and not a min_count)
    print(" ")
    print("Loading images from the directory '." + directory + "'.")
    print(" ")
    # Init lists
    images = []
    labels = []
    cls = []
    filenames = []
    
    # Alphabetically sorted classes
    class_dirs = sorted(glob.glob(directory + "/*"))
    # Load images from each class
    idx = 0 #Set class ID
    for class_dir in class_dirs:

        # Class name
        class_name = os.path.basename(class_dir)
        if class_name in classes_select:
        
            num_files = len(os.listdir(class_dir))
            print("%s - %d" % (class_name, num_files))
            n_samples=0
            class_idx = idx
            idx += 1                                    #Increment class ID
            labels.append(class_name)                   #Add current class label to labels master list
            
            # Get the files
            files = sorted(glob.glob(class_dir + "/*." + extension))
            random.Random(choose_random).shuffle(files)
                
            for file in files:
                if n_samples > max_samples: continue
                n_samples +=1
                im = process_image(file, width)
                if norm:
                    im = rescale_intensity(im, in_range='image', out_range=(0.0, 1.0))
                images.append(im)                       #Add current image to images array
                cls.append(class_idx)                   #Add current image label to cls array
                filenames.append(os.path.basename(os.path.dirname(file)) + os.path.sep + os.path.basename(file))
            print(n_samples)

    # Final clean up
    images = np.asarray(images) #training_images
    cls = np.asarray(cls)       #training_labels

    return images, cls, labels, filenames



#%%
 
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
   
    """
    This function prints and plots the confusion matrix. Normalization can be 
    applied by setting `normalize=True`.

    Parameters
    ----------
    cm : Numpy array
        The confusion matrix data
    classes : list
        The different labels.
    normalize : bool, optional
        Plot a normalized-version of the data or not; The default is False.
    title : string, optional
        Plot title. The default is 'Confusion matrix'.
    cmap : plt.cm object, optional
        Color mapping. The default is plt.cm.Blues.

    Returns
    -------
    None.

    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#%%
def compute_accuracy (test_images, test_labels, predict_data):
    
    """
    This function calculates the model's accuracy and prints it out.
    
    Returns
    -------
    The accuracy metrics so it can be added to the post-training notes

    """
    
    
    success = 0
    fail = 0

    
    for b in range (test_images.shape[0]):

        if test_labels[b] == predict_data [b]:
          success =  success +1
        else:
          fail = fail +1    
     
    prob_success=round(success/len(test_labels)*100,1)
    prob_failure=round(fail/len(test_labels)*100,1)
    
    print(" ")  
    print('Pollen successfully classified  : {0}% ({1}/{2})'.format(prob_success, success,len(test_labels)))
    print('Pollen misclassified : {0}% ({1}/{2})'.format(prob_failure, fail,len(test_labels)))
    print('Total under threshold : '+str(under_threshold))

        
    return success, fail, prob_success, prob_failure

# %%

def count_argmax_under_threshold(prediction_data, threshold):

  max_predictions = np.max(prediction_data, axis=1)
  count = np.sum(max_predictions < threshold)
  
  return count


#%%
def temp_scaling(logits_nps, labels_nps, sess, maxiter=50):
    """
    
    Calculates, using the NLL, the temperature value used to calibrate the predictions.
    Since we're using tf1.14, the eager execution has to be disabled before 
    calling this function.
    Adapted from https://github.com/markdtw/temperature-scaling-tensorflow    

    Parameters
    ----------
    logits_nps : Numpy array of float32
        Predictions. Has to be float32
        
    labels_nps : Array of int
        Testing labels.
        
    sess : Tensorflow object
        Permits step-by-step manipulation of the training process by the function
        
    maxiter : Int, optional
        Max iterations to minimize the NLL. The default is 50.

    Returns
    -------
    temperature : Int
        The value (>0) by which the data output is calibrated according according
        to Guo et al. 2017.
        
    scld_predict : Numpy array
        The calibrated predictions.
        
    scld_per : Numpy array
        The calibrated predictions (percent).

    """


    temp_var = tf.get_variable("temp", shape=[1], initializer=tf.initializers.constant(1.5))
    
    
    labels_tensor = tf.constant(labels_nps, name='labels_valid')
    
    
    
    #Functions differently depending on the activation function and loss function chosen during model training
    #It is necessary to reshape the logits tensor to make it fit into the function
    
    if ac_function == 'sigmoid':
        logits_nps_1d = logits_nps.flatten()
        logits_tensor = tf.constant(logits_nps_1d, name='logits_valid')
        acc_op = tf.metrics.accuracy(labels_tensor, logits_tensor)
        logits_w_temp = tf.divide(logits_tensor, temp_var)
        nll_loss_op = tf.keras.losses.binary_crossentropy(
            y_true=labels_tensor, y_pred=logits_w_temp, from_logits=True)
        
    else: 
        logits_tensor = tf.constant(logits_nps, name='logits_valid')
        acc_op = tf.metrics.accuracy(labels_tensor, tf.argmax(logits_tensor, axis=1))
        logits_w_temp = tf.divide(logits_tensor, temp_var)
        nll_loss_op = tf.losses.sparse_softmax_cross_entropy(
            labels= labels_tensor, logits=logits_w_temp)

    org_nll_loss_op = tf.identity(nll_loss_op)

    # optimizer
    print("CHOOSE OPTIMIZER")
    optim = tf.contrib.opt.ScipyOptimizerInterface(nll_loss_op, options={'maxiter': maxiter})

    sess.run(temp_var.initializer)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    org_nll_loss = sess.run(org_nll_loss_op)

    optim.minimize(sess)

    nll_loss = sess.run(nll_loss_op)
    temperature = sess.run(temp_var)
    acc = sess.run(acc_op)

    print ("Original NLL: {:.3f}, validation accuracy: {:.3f}%".format(org_nll_loss, acc[0] * 100))
    print ("After temperature scaling, NLL: {:.3f}, temperature: {:.3f}".format(
        nll_loss, temperature[0]))
    

    predict_w_temp = logits_w_temp.eval(session=sess)
    
    scld_predict = np.exp(predict_w_temp) / np.sum(np.exp(predict_w_temp),
                                                     axis=-1, keepdims=True)
    

    if ac_function == 'sigmoid': 
        i = 0
        scld_per = np.zeros(shape=(len(scld_predict),), dtype='float32')
        for value in scld_predict:
            if value > 0:
                scld_per[i] = value*1./max(scld_predict)
            else:
                scld_per[i] = value
            i = i+1
            
    else:
        
        scld_per = np.where(np.max(scld_predict, axis=0)==0, scld_predict,
                                      scld_predict*1./np.max(scld_predict, axis=0))


    return temperature, scld_predict, scld_per
    
# %%

# =============================================================================
# Main Script
# =============================================================================

# Load the images
images, cls, labels, filenames = load_from_class_dirs(path_folder_image, "png", resolution, False)

#%%
# ======================
# Splitting full dataset
# ======================


# Split images (var images), labels (var cls) and filenames (var filenames) into train set and test set


# Training/testing split
train_val_images, test_images, train_val_labels, test_labels, train_val_filenames, \
    test_filenames = train_test_split(images, cls, filenames, test_size=0.15, random_state=choose_random)

# Training/validation split
train_images, val_images, train_labels, val_labels, train_filenames, \
    val_filenames = train_test_split(train_val_images, train_val_labels, train_val_filenames, test_size=0.1, random_state=choose_random)



# Optimizer
  
opt = tf.keras.optimizers.Adam()


print("Images splitted; optimizer chosen")
#%%
# ======================
# Data augmentation
# ======================

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                               fill_mode= 'constant', cval = 0, shear_range = 0.15,
                                #brightness_range = [0.8, 1.0,],
                              rotation_range = 45
                             )
datagen.fit(train_images)


print("Data augmentation generator fitted to training data")
# %%
# ======================
# Transfer learning
# ======================


  
base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

# Pre-trained weights from imagenet are loaded for transfer learning
# include_top = false -> we do not include the fully connected hear with the softmax classifier
# The forward propagation stops at the max-pooling layer - We will treat the 
# output of the max-pooling layer as a list of features, also known as a feature vector

#Lock layers
for layer in base_model.layers[0:3]:
    layer.trainable = False

base_model.summary()


#%%
# =====================
# Building the model
# =====================

model = tf.keras.models.Sequential()

model.add(base_model.layers[3])

model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D())

if use_deep_model == True:
    model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5,seed=7))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    
else:
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5,seed=7))
    model.add(tf.keras.layers.Dense(512, activation='relu'))

model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))


# model.count_params()
if ac_function == 'sigmoid':
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
else:
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# Loss function: Measures how accurate the model is during training. You want to minimize this function to 'steer' the model in its right direction        
# Optimizer: How the model is updated based on the data it sees and its loss function
# Metrics: Used to monitor the training and testing steps. 'accuracy' = fraction of the images that are correctly classified



print("Keras layers compiled")

#%%

# ======================
# Create and assign callbacks and start training
# ======================

# Load the checkpoint callback; Added to the model if will_save == True
# Saves the weights during training
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, \
                                                 verbose = 1, period = 5) #period = save frequency

# Load the Reduce Learning Rate on Plateau callback;
# Allows to further train the model if the validation loss metric plateaus
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, \
                                                   patience=20, verbose = 1, mode='auto', \
                                                   min_delta=0.001, cooldown=5, min_lr=0)

# Load the early stopping callback;
# Stops training if the validation loss metric hasn't increased by min_delta in epoch = patience
es_callback = tf.keras.callbacks.EarlyStopping(patience = 100, monitor = 'val_loss', verbose = 1, min_delta = 0.0001, 
                                               restore_best_weights=True, mode='auto')

# Add the desired callbacks
callbacks = [lr_callback, es_callback]
if will_save == True: 
    callbacks.append(cp_callback)
    if cp_callback.period > n_epochs:
            print("The model will train but no checkpoints will be saved - \n"
          "Consider lowering callback period value.")


# Fit the model/start training
# If will_train is set to false - will load the a local checkpoint instead.
# The loaded weights are either from the latest checkpoint of from the 
# one specified in the parameters section.


if will_train == True:
    
    print('TRAINING')

    pollen_cnn = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size), 
                                      steps_per_epoch=len(train_images) / batch_size, epochs=n_epochs, 
                                        callbacks=callbacks, \
                                            validation_data = (val_images, val_labels))
    
    # Plot the training graph    
    loss = pollen_cnn.history['loss']
    accuracy = pollen_cnn.history['acc']
    plt.plot(loss)
    plt.plot(accuracy)
    plt.legend(['loss', 'accuracy'])
    
    # Save the graph and the model in the training folder
    if will_save == True: 
        plt.savefig(os.path.dirname(checkpoint_path)+"/training_graph.png")
        model.save(os.path.dirname(checkpoint_path)+"/"+checkpoint_no+"_model")
        print("Model saved at "+checkpoint_path+"/"+checkpoint_no+"_model")
        
    plt.show()
    print("Training graph plotted")
    


else:


    if load_weights == True:
        print("LOADING CHECKPOINT " + latest)
        pollen_cnn  = model.load_weights(latest)
        
    else:
        print("Model loaded")
        model = tf.keras.models.load_model(os.path.dirname(checkpoint_path)+"/"
                                           +checkpoint_no+"_model")

#%%

# ======================
# Test model accuracy on the test dataset
# ======================


if ac_function == 'sigmoid':    
    predictions = (model.predict(test_images) >= 0.5).astype("int64")
    predictions = predictions[0:,0] 

else: predictions = (model.predict_classes(test_images))[0:test_images.shape[0]]
    

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) 

under_threshold = count_argmax_under_threshold(model.predict(test_images), threshold=threshold)

success_images, fail_images, p_success, p_fail = compute_accuracy(test_images, test_labels, predictions)

print("Test accuracy:", test_acc)

if will_train == True:
    print("Min. training loss : "+str(min(loss))+" at epoch n "+str(1+(loss.index(min(loss)))))
    print("Max accuracy : "+str(max(accuracy))+" at epoch n "+str(1+(accuracy.index(max(accuracy)))))
    print('Total under threshold : '+str(under_threshold))


# Save Notes in checkpoint folder
if will_save == True and will_train == True:
    notes_ckpt = notes_ckpt+"\nCallbacks used : "+str(callbacks)+\
        "\nPollen successuffully classified  : {0}% ({1}/{2})".format(p_success, success_images,len(test_labels))+\
        "\nPollen misclassified : {0}% ({1}/{2})".format(p_fail, fail_images,len(test_labels))+\
            "\nTest accuracy : "+str(test_acc)+"\nMin. training loss : "+str(min(loss))+\
                " at epoch n "+str(1+(loss.index(min(loss))))+"\nMax accuracy : "+\
                    str(max(accuracy))+" at epoch n "+str(1+(accuracy.index(max(accuracy))))
    file1= open("checkpoints_saves/"+checkpoint_no+"/"+"notes_"+checkpoint_no+".txt", "w")
    file1.write(notes_ckpt)
    file1.close()
    print("Training notes saved") 


#%%

# ======================
# Confusion matrix
# ======================

y_pred = predictions   # y pred = predicted labels
cnf_matrix = confusion_matrix(test_labels, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, labels, title='Confusion matrix, without normalization')
lbl_positions = range(len(labels))


plt.xticks(lbl_positions, classes_select)
plt.yticks(lbl_positions, classes_select)
    
if will_save == True: plt.savefig(os.path.dirname(checkpoint_path)+"/non_normalized_confusion_matrix.png")

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, labels, normalize=True, title='Normalized confusion matrix')
lbl_positions = range(len(labels))

   

plt.xticks(lbl_positions, classes_select)
plt.yticks(lbl_positions, classes_select)

if will_save == True: plt.savefig(os.path.dirname(checkpoint_path)+"/normalized_confusion_matrix.png")
    
plt.show()



#%%

# ======================
# Calibration
# ======================

# Predict on the test images (this time using model.predict)
# model.predict_classes predicts the class (as an int)

predict_non_int = model.predict(val_images)

# Disable eager execution for compatibility; reset the graph
# You will have to restart the kernel before running the script again

tf.disable_eager_execution()
tf.reset_default_graph()

temperature, scld_predict, scld_per = temp_scaling(predict_non_int, 
                                                  val_labels, tf.Session(), maxiter=50)

# Generate an array of the predicted classes
if ac_function == 'sigmoid':
    scld_int = (scld_per >= 0.5).astype("int64")
else:
    scld_int = np.argmax(scld_per, axis = 1)


# Compute the calibrated accuracy using a threshold
success_images, fail_images, p_success, p_fail = compute_accuracy(val_images, 
                                                       val_labels, scld_int)










