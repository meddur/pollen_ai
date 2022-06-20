# =============================================================================
# CNN Bela
# =============================================================================

import tensorflow as tf
tf.enable_eager_execution() #Parce que TF version <2.0 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16 #utilisent RESNET_18 Olsson et al 
from tensorflow.keras import Model
#from tensorflow.keras.applications.vgg16 import preprocess_input




from sklearn.calibration import calibration_curve




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
                        # "pinus_mix",
                    # "pinus_s",
                    # "populus_d",
                    # "quercus_r",
                    # 'tricolp_mix',
                    # 'tripor_mix',
                    # 'tsuga',
                    #'vesiculate_mix',
                    # "bidon1",
                    # "bidon2"
                    
        ]                                   # Classes you want (or not) in the model

pretty_labels= False

lbl_pretty = (
                    "abb_pic_mix",
                    "abies_b",
                    "acer_mix",
                    "acer_r",
                    "acer_s",
                    'alnus_mix',
                    "alnus_c",
                    "alnux_r",
                    "betula_mix",
                    "corylus_c",
                    "eucalyptus",
                    "juni_thuya",
                    "picea_mix",
                    "pinus_b_mix",
                    "pinus_mix",
                    "pinus_s",
                    "populus_d",
                    "quercus_r"
    )

add_unclass = False



#Name of the directory


os.chdir("/Users/mederic/Documents/python/CNN_BELA")

if add_unclass == True: path_folder_image = "/home/paleolab/Documents/python/CNN_BELA/pollen_dataset_w_unclass"
else: path_folder_image = os.getcwd()+"/pollen_dataset/level_0"



#Parameters

max_samples = 1200
plot_that_shit = False
n_epochs = 600
choose_random = 20
ac_function = "softmax" #Don't forget to change the loss function to binary_crossentropy 
batch_size = 32

simple_model = False

#Checkpoint setup

will_train = False       # False = load an already existing model
will_save = False



checkpoint_no ="transfer_tripor_small_deep"
checkpoint_path = "checkpoints/"+checkpoint_no+"/cp-{epoch:04d}.ckpt"


latest = tf.train.latest_checkpoint("checkpoints/"+checkpoint_no)
# latest = ("checkpoints/"+checkpoint_no+"/cp-0280.ckpt") #Checkpoint en particulier?

#########################################################################

# os.path.exists(os.getcwd()+"checkpoints/"+checkpoint_no)

if will_save == True and os.path.exists(os.getcwd()+"/checkpoints/"+checkpoint_no) == True:

    overwrite_check = input("Checkpoint already exists. Overwrite (y/n)?")
    if overwrite_check != "y": 
        sys.exit('Error : Checkpoint folder already exists. Aborting')
    else:
        directory_wipe = glob.glob(os.getcwd()+"checkpoints/"+checkpoint_no+"/*")
        for f in directory_wipe:         
            os.remove(f)
    

notes="test_bidon_ece"


notes_ckpt = "Checkpoint : "+checkpoint_no+"\nResolution : " + str(resolution) + \
    "\nClasses : " + str(classes_select) + "\nMax samples : " + str(max_samples) + \
        "\nEpochs : " + str(n_epochs) + "\nAdd Unclass : " + str(add_unclass) + \
            "\nActivation function : " + ac_function + \
                "\nSimple model : " + str(simple_model) + \
            "\nSeed : " + str(choose_random) + "\nNotes : "+ notes + "\n Path folder image : "+path_folder_image

print(notes_ckpt)



if ac_function == "softmax" and len(classes_select) == 2: print("SOFTMAX FUNCTION USED FOR BINARY CLASSIFICATION")
elif ac_function == "sigmoid" and len(classes_select) > 2: print("SIGMOID FUNCTION USED FOR NON-BINARY CLASSIFICATION")


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
        
        if (class_name in classes_select) == presence:  # Import (or not) classes included in classes_select
            num_files = len(os.listdir(class_dir))
            print("%s - %d" % (class_name, num_files))
            n_samples=0
            if num_files < min_count: continue # ?

            class_idx = idx
            idx += 1                                    #Increment class ID
            labels.append(class_name)                   #Add current class label to labels master list
            
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
                cls.append(class_idx)                   #Add current image label to cls array
                filenames.append(os.path.basename(os.path.dirname(file)) + os.path.sep + os.path.basename(file))
            print(n_samples)

    # Final clean up
    images = np.asarray(images) #training_images
    cls = np.asarray(cls)       #training_labels
    num_classes = len(labels)   #Amount of classes
    return images, cls, labels, num_classes, filenames


#%%

#Definition of plot functions
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])  
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'  
  plt.xlabel("{} {:2.0f}% ({})".format(labels[predicted_label],
                                100*np.max(predictions_array),
                                labels[true_label]),
                                color=color)
  plt.ylabel(test_filenames[i][0:len(test_filenames[i])-4])
#%%
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks(range(num_class), labels, rotation=45)       # range(total amount of classes)
  plt.yticks([])
  thisplot = plt.bar(range(num_class), predictions_array, color="#777777")      # range
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array) 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
#%%
# This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
def temp_scaling(logits_nps, labels_nps, sess, maxiter=50):
    #Stolen from https://github.com/markdtw/temperature-scaling-tensorflow

    temp_var = tf.get_variable("temp", shape=[1], initializer=tf.initializers.constant(1.5))
    
    
    labels_tensor = tf.constant(labels_nps, name='labels_valid')
    
    
    
    #Functions differently depending on the activation function and loss function chosen during training
    #It is necessary to reshape the logits tensor to make it fit into the function
    
    if ac_function == 'sigmoid':     #ADDED THIS
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
    sess.run(tf.global_variables_initializer()) #ADDED THIS
    org_nll_loss = sess.run(org_nll_loss_op)

    optim.minimize(sess)

    nll_loss = sess.run(nll_loss_op)
    temperature = sess.run(temp_var)
    acc = sess.run(acc_op)

    print ("Original NLL: {:.3f}, validation accuracy: {:.3f}%".format(org_nll_loss, acc[0] * 100))
    print ("After temperature scaling, NLL: {:.3f}, temperature: {:.3f}".format(
        nll_loss, temperature[0]))
    

    #ADDED THIS###########
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

    #####################
    return temperature, scld_predict, scld_per


# scld_per = np.where(np.max(predict_non_int, axis=0)==0, predict_non_int,
#                               predict_non_int*1./np.max(predict_non_int, axis=0))

    
# %%

# =============================================================================
# Main Script
# =============================================================================

#Extract images
images, cls, labels, num_classes, filenames = load_from_class_dirs(path_folder_image, "png", resolution, False, min_count=20)
#Changed the extension to png, got an error -> im = resize var mode
#I don't think leaving the extension as .tif did anything

#images = images[:,:,:,np.newaxis] #4th dimension fix

print("reshape")


#%%



#Split images (var images), labels (var cls) and filenames (var filenames) into train set and test set

# train_images, test_images, train_labels, test_labels, train_filenames, \
#     test_filenames = train_test_split(images, cls, filenames, test_size=0.15, random_state=choose_random)

###TRAIN TEST SPLIT###
train_val_images, test_images, train_val_labels, test_labels, train_val_filenames, \
    test_filenames = train_test_split(images, cls, filenames, test_size=0.15, random_state=choose_random)

###TRAIN VAL SPLIT###
train_images, val_images, train_labels, val_labels, train_filenames, \
    val_filenames = train_test_split(train_val_images, train_val_labels, train_val_filenames, test_size=0.1, random_state=choose_random)



# Optimizer
###############################################################################
  
opt = tf.keras.optimizers.Adam()


print("Images splitted; optimizer")
#%%


datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                               fill_mode= 'constant', cval = 0, shear_range = 0.15,
                                #brightness_range = [0.8, 1.0,],
                                # zoom_range = [0.7, 1.0],
                               # height_shift_range = 0.1, width_shift_range=0.1,
                              rotation_range = 45,
                             )
datagen.fit(train_images)


# %%
# ======================
# Transfer learning
# ======================

print("Loading Network")


# base_model = VGG16(input_shape = (resolution, resolution, 3), weights = "imagenet", include_top=False)  
base_model = tf.keras.applications.VGG16(input_shape = (resolution, resolution, 3),
                                          include_top = False,
                                          weights = 'imagenet')

#Pre-trained weights from ImageNet are loaded for transfer learning
# include_top = false -> we do not include the fully connected hear with the softmax classifier
# The forward propagation stops at the max-pooling layer - We will treat the output of the max-pooling layer as a list of features, also known as a feature vector

#Lock layers
for layer in base_model.layers[0:6]:
    layer.trainable = False

base_model.trainable = False

#connect one layer from the CNN to our CNN
#please experiment

#Compile the model

# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
base_model.summary()



# base_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#%%



    #16 -> Number of output filters in the convolution
    #(3,3) -> Kernel size (Height and Width of the 2d convolution window)
    #padding = same -> for each filter (16,32 etc) there is an output channel

if simple_model == True :
  model = tf.keras.models.Sequential()
  # model.add(base_model.layers[0])
  # model.add(base_model.layers[1])
  # model.add(base_model.layers[2])
  model.add(base_model.layers[3])
  model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())                                        
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dropout(0.5,seed=7))
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))    


elif resolution == 64 :

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
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dropout(0.5,seed=7))
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))
  

elif resolution == 128 :
    
    # model = tf.keras.models.Sequential([
    #     base_model,
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(len(labels), activation=ac_function)
    #     ])

    model = tf.keras.models.Sequential()
    # model.add(base_model.layers[0])
    # model.add(base_model.layers[1])
    # model.add(base_model.layers[2])
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
    if ac_function == 'sigmoid':
        model.add(tf.keras.layers.Dense(1, activation=ac_function)) #THIS LAYER SHOULD HAVE FILTER SIZE 1 IF BINARY CLASS.
        # model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))
    else:
        model.add(tf.keras.layers.Dense(len(labels), activation=ac_function)) #THIS LAYER SHOULD HAVE FILTER SIZE 1 IF BINARY CLASS.


# model.count_params()
if ac_function == 'sigmoid':
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
else:
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#         # Loss function: Measures how accurate the model is during training. You want to minimize this function to 'steer' the model in its right direction        
#         # Optimizer: How the model is updated based on the data it sees and its loss function
#         # Metrics: Used to monitor the training and testing steps. 'accuracy' = fraction of the images that are correctly classified

# model.build(input_shape = (0,128, 128, 3))
# model.summary()

print("keras layers compiled")

#%%

#Create callbacks

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, \
                                                 verbose = 1, period = 10) #period = save frequency

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, \
                                                   patience=20, verbose = 1, mode='auto', \
                                                   min_delta=0.001, cooldown=5, min_lr=0)


es_callback = tf.keras.callbacks.EarlyStopping(patience = 150, monitor = 'val_loss', verbose = 1, min_delta = 0.0001, 
                                               restore_best_weights=True, mode='auto')

# callbacks = []
callbacks = [es_callback]
if will_save == True: callbacks.append(cp_callback)


#Fit the model/start training

if will_train == True:
    
    print('TRAINING')

    pollen_cnn = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size), 
                                      steps_per_epoch=len(train_images) / batch_size, epochs=n_epochs, 
                                        callbacks=callbacks, \
                                            validation_data = (val_images, val_labels))
        
    loss = pollen_cnn.history['loss']
    accuracy = pollen_cnn.history['acc']
    plt.plot(loss)
    plt.plot(accuracy)
    plt.legend(['loss', 'accuracy'])
    if will_save == True: plt.savefig(os.path.dirname(checkpoint_path)+"/training_graph.png")
    plt.show()
    print("Loss+accuracy plotted")

else:

    print("LOADING CHECKPOINT " + latest)
    pollen_cnn = model.load_weights(latest)    


#%%
#Test the model w/ predictions on the test set
#BINARY CLASSIFICATION: ADDED NEW METHOD

if ac_function == 'sigmoid':    
    sonic = (model.predict(test_images) >= 0.5).astype("int64")
    sonic = sonic[0:,0] 

else: sonic = (model.predict_classes(test_images))[0:test_images.shape[0]]
    

    # Tests are processed faster when predictions are in here
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) 

success_images  = 0
fail_images = 0  

# if ac_function == 'sigmoid':    #Binary classification // model.predict_classes is deprecated
     
# else:
for b in range (test_images.shape[0]):
    if test_labels[b] == sonic [b]:
      success_images =  success_images +1
    else:
      fail_images = fail_images +1

    
p_success= round(success_images/len(test_labels)*100,1)
p_fail=round(fail_images/len(test_labels)*100,1)
    
print(" ")  
print('Pollen successuffully classified  : {0}% ({1}/{2})'.format(p_success, success_images,len(test_labels)))
print('Pollen misclassified : {0}% ({1}/{2})'.format(p_fail, fail_images,len(test_labels)))


print("Test accuracy:", test_acc)
if will_train == True:
    print("Min. training loss : "+str(min(loss))+" at epoch n "+str(1+(loss.index(min(loss)))))
    print("Max accuracy : "+str(max(accuracy))+" at epoch n "+str(1+(accuracy.index(max(accuracy)))))
    



#%%
#Save Notes
if will_save == True and will_train == True:
    notes_ckpt = notes_ckpt+"\nCallbacks used : "+str(callbacks)+\
        "\nPollen successuffully classified  : {0}% ({1}/{2})".format(p_success, success_images,len(test_labels))+\
        "\nPollen misclassified : {0}% ({1}/{2})".format(p_fail, fail_images,len(test_labels))+\
            "\nTest accuracy : "+str(test_acc)+"\nMin. training loss : "+str(min(loss))+\
                " at epoch n "+str(1+(loss.index(min(loss))))+"\nMax accuracy : "+\
                    str(max(accuracy))+" at epoch n "+str(1+(accuracy.index(max(accuracy))))
    file1= open("checkpoints/"+checkpoint_no+"/"+"notes_"+checkpoint_no+".txt", "w")
    file1.write(notes_ckpt)
    file1.close()
    print("Model saved") 

#%%

# =============================================================================
# Graph that shit
# =============================================================================


predictions = model.predict(test_images)
num_class = len(labels)
num_rows = 200
num_cols = 3
num_images = num_rows*num_cols
test_images2d=test_images[:,:,:,0]    

if plot_that_shit == True:
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    i=0
    a=0
    while a < num_images:
        if i < test_images.shape[0]:   # here "if" function is security for the "loob for" if there are too many test images
            if test_labels[i] != sonic[i]:
                plt.subplot(num_rows, 2*num_cols, 2*a+1)
                plot_image(i, predictions, test_labels, test_images2d)
                plt.subplot(num_rows, 2*num_cols, 2*a+2)
                plot_value_array(i, predictions, test_labels)
            else:
                a=a-1
        else:
            break
        i=i+1
        a=a+1

    plt.show()
    
#%%
#Confusion matrix
y_pred = sonic   # y pred = predicted labels
cnf_matrix = confusion_matrix(test_labels, y_pred, labels=range(num_classes))   #"labels=" List of labels to index the matrix. This may be used to reorder or select a subset of labels. If none is given, those that appear at least once in y_true or y_pred are used in sorted order.
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, labels, title='Confusion matrix, without normalization')
lbl_positions = range(len(labels))

if pretty_labels == True:        
    plt.xticks(lbl_positions, lbl_pretty)
    plt.yticks(lbl_positions, lbl_pretty)
else:
    plt.xticks(lbl_positions, classes_select)
    plt.yticks(lbl_positions, classes_select)
    
if will_save == True: plt.savefig(os.path.dirname(checkpoint_path)+"/non_normalized_confusion_matrix.png")

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, labels, normalize=True, title='Normalized confusion matrix')
lbl_positions = range(len(labels))

if pretty_labels == True:
    plt.xticks(lbl_positions, lbl_pretty)
    plt.yticks(lbl_positions, lbl_pretty)
   
else:
    plt.xticks(lbl_positions, classes_select)
    plt.yticks(lbl_positions, classes_select)

if will_save == True: plt.savefig(os.path.dirname(checkpoint_path)+"/normalized_confusion_matrix.png")
    
plt.show()

#%%

predict_non_int = model.predict(test_images)

sys.path.insert(0, '/Users/mederic/Documents/python/temperature-scaling-tensorflow-master')
tf.disable_eager_execution()
# tf.reset_default_graph()

# import temp_scaling
#%%

predict_non_int = model.predict(test_images)


tf.disable_eager_execution()
tf.reset_default_graph()

temperature, scld_predict, scld_per = temp_scaling(predict_non_int, 
                                                  test_labels, tf.Session(), maxiter=50)

##############################
#ACCURACY POST SCALING
##############################

if ac_function == 'sigmoid':
    scld_int = (scld_per >= 0.5).astype("int64")
else:
    scld_int = np.argmax(scld_per, axis = 1)

success_images = 0
fail_images = 0

for b in range (test_images.shape[0]):
    if test_labels[b] == scld_int [b]:
      success_images =  success_images +1
    else:
      fail_images = fail_images +1

p_success=round(success_images/len(test_labels)*100,1)
p_fail=round(fail_images/len(test_labels)*100,1)
    
print(" ")  
print('Pollen successuffully classified  : {0}% ({1}/{2})'.format(p_success, success_images,len(test_labels)))
print('Pollen misclassified : {0}% ({1}/{2})'.format(p_fail, fail_images,len(test_labels)))
