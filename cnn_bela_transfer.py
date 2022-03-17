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

from tensorflow.keras.applications import VGG16 #utilisent RESNET_18 Olsson et al 
from tensorflow.keras import Model
#from tensorflow.keras.applications.vgg16 import preprocess_input





print("TensorFlow version " + tf.__version__)

# =============================================================================
# Settings
# =============================================================================

resolution = 128              
presence = True                # True = classes included in classes_select WILL be included in the model - False = WILL NOT be included
classes_select = [
                     "abb_pic_mix",
                   # "abies_b",
                    #"acer_mix",
                    # "acer_r",
                    # "acer_s",
                     'alnus_mix',
                    # "alnus_c",
                    # "alnus_r",
                      "betula_mix",
                       "corylus_c",
                       "eucalyptus",
                       "juni_thuya",
                    # "picea_mix",
                    # "pinus_b_mix",
                     "pinus_mix",
                    # "pinus_s",
                    # "populus_d",
                    #"quercus_r",
                    'tricolp_mix',
                    #'vesiculate_mix',
                    
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

if add_unclass == True: path_folder_image = "/home/paleolab/Documents/python/CNN_BELA/pollen_dataset_w_unclass"
else: path_folder_image = "/home/paleolab/Documents/python/CNN_BELA/pollen_dataset/level_0"


#Parameters

max_samples = 60
plot_that_shit = False
n_epochs = 10
choose_random = 20
ac_function = "softmax"
batch_size = 32

simple_model = True

#Checkpoint setup

will_train = True        # False = load an already existing model
will_save = True


checkpoint_no ="test"
checkpoint_path = "checkpoints/"+checkpoint_no+"/cp-{epoch:04d}.ckpt"
#checkpoint_path = checkpoint_no+"/cp-0185.ckpt"



if will_save == True and os.path.exists(os.getcwd()+"checkpoints/"+checkpoint_no) == True:

    overwrite_check = input("Checkpoint already exists. Overwrite (y/n)?")
    if overwrite_check != "y": 
        sys.exit('Error : Checkpoint folder already exists. Aborting')
    else:
        directory_wipe = glob.glob(os.getcwd()+"checkpoints/"+checkpoint_no+"/*")
        for f in directory_wipe:         
            os.remove(f)
    
notes=" doubled starting outputfilters (16->32etc)"

notes_ckpt = "Checkpoint : "+checkpoint_no+"\nResolution : " + str(resolution) + \
    "\nClasses : " + str(classes_select) + "\nMax samples : " + str(max_samples) + \
        "\nEpochs : " + str(n_epochs) + "\nAdd Unclass : " + str(add_unclass) + \
            "\nActivation function : " + ac_function + \
                "\nSimple model : " + str(simple_model) + \
            "\nSeed : " + str(choose_random) + "\nNotes : "+notes 
                

print(notes_ckpt)



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
train_images, test_images, train_labels, test_labels, train_filenames, \
    test_filenames = train_test_split(images, cls, filenames, test_size=0.15, random_state=choose_random)

    #removed cls_mapping split because I haven't figured out what it does yet
    #random_state = int -> CHANGE THIS
    #test_size = float. proportion accordée au test sample



# Optimizer
###############################################################################
  
opt = tf.keras.optimizers.Adam()


print("Images splitted; optimizer")


datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                               fill_mode= 'constant', cval = 0, shear_range = 0.15,
                                #brightness_range = [0.8, 1.0,],
                               # zoom_range = [0.7, 1.0],
                              # height_shift_range = 0.1, width_shift_range=0.1, shear_range = 0.15,
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



print(base_model.summary())

#Pre-trained weights from ImageNet are loaded for transfer learning
# include_top = false -> we do not include the fully connected hear with the softmax classifier
# The forward propagation stops at the max-pooling layer - We will treat the output of the max-pooling layer as a list of features, also known as a feature vector

#Lock layers
for layer in base_model.layers:
    layer.trainable = False


#connect one layer from the CNN to our CNN
#please experiment

#Add a classificaiton head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(labels), activation = 'softmax') #Try sigmoid
    ])

#Compile the model

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()




#Input shape de la première layer du modèle normal = (None, 128, 128, 1)

# #Add a dropout rate
# prediction_model = tf.keras.layers.Dropout(0.2)(prediction_model)

# #Add a final softmax layer for classification
# prediction_model = tf.keras.layers.Dense (3, activation = ac_function)(prediction_model)



# model = Model(pre_trained_model.input, prediction_model)







#%%



    #16 -> Number of output filters in the convolution
    #(3,3) -> Kernel size (Height and Width of the 2d convolution window)
    #padding = same -> for each filter (16,32 etc) there is an output channel
# if simple_model == True :
#   model = tf.keras.models.Sequential()
#   model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())
#   model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())
#   model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())
#   model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())
#   model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())                                        
#   model.add(tf.keras.layers.Flatten())
#   model.add(tf.keras.layers.Dropout(0.5,seed=7))
#   model.add(tf.keras.layers.Dense(512, activation='relu'))
#   model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))    

# elif resolution == 64 :
#   model = tf.keras.models.Sequential()
#   model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
#   model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())
#   model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())
#   model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())
#   model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
#   model.add(tf.keras.layers.MaxPooling2D())
#   model.add(tf.keras.layers.Flatten())
#   model.add(tf.keras.layers.Dropout(0.5,seed=7))
#   model.add(tf.keras.layers.Dense(512, activation='relu'))
#   model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))
  
if resolution == 128 :
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
   model.add(tf.keras.layers.Dense(len(labels), activation=ac_function))



# model.count_params()
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#         # Loss function: Measures how accurate the model is during training. You want to minimize this function to 'steer' the model in its right direction        
#         # Optimizer: How the model is updated based on the data it sees and its loss function
#         # Metrics: Used to monitor the training and testing steps. 'accuracy' = fraction of the images that are correctly classified

# model.summary()

print("keras layers compiled")

#%%

#Create a callback that saves the models weights

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, \
                                                 verbose = 1, period = 10) #period = save frequency

    
#Fit the model/start training

if will_train == True:
    if will_save == True:
        print('TRAINING AND SAVING')
        pollen_cnn = model.fit(train_images, train_labels, epochs=n_epochs, batch_size = batch_size, callbacks=[cp_callback])
    else:
        print('TRAINING')
        pollen_cnn = model.fit(train_images, train_labels, epochs=n_epochs, batch_size = batch_size)
    loss = pollen_cnn.history['loss']
    accuracy = pollen_cnn.history['acc']
    plt.plot(loss)
    plt.plot(accuracy)
    plt.legend(['loss', 'accuracy'])
    if will_save == True: plt.savefig(os.path.dirname(checkpoint_path)+"/training_graph.png")
    plt.show()
    print("Loss+accuracy plotted")

else:
    latest = tf.train.latest_checkpoint(checkpoint_no)
    # latest = (checkpoint_no+"/cp-0190.ckpt") #Checkpoint en particulier?
    print("LOADING CHECKPOINT " + latest)
    pollen_cnn = model.load_weights(latest)
    


#%%

# loss = pollen_cnn.history['loss']
# accuracy = pollen_cnn.history['acc']
# plt.plot(loss)
# plt.plot(accuracy)
# plt.legend(['loss', 'accuracy'])
# plt.show()

# print("Loss+accuracy plotted")

#%%
#Test the model w/ predictions on the test set
sonic=(model.predict_classes(test_images))[0:test_images.shape[0]]
    # Tests are processed faster when predictions are in here
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) 

success_images  = 0
fail_images = 0  
for b in range (test_images.shape[0]):
  if test_labels[b] == sonic[b]:
    success_images =  success_images  +1
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
    notes_ckpt = notes_ckpt+"\nPollen successuffully classified  : {0}% ({1}/{2})".format(p_success, success_images,len(test_labels))+\
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


