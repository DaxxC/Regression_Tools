import tensorflow as tf
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras import  Sequential
import zipfile
import os
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime

def model_early_stopping_callback(metric='val_loss', delta=0,patience=0, best_weights=False):
  """
  Returns a Tensorflow callback for early stopping of a model in training. Metric is the
  monitored value to cause the model to stop training. It is val_loss by default. Delta is how 
  much the metric much it must change by before it stops. Pateince is the number of epochs it must 
  must not have met the threshold by. Best weights is whether or not the best weights will e restored.
  """
  
  return tf.keras.callbacks.EarlyStopping(
      monitor= metric,
      min_delta=delta ,
      patience=0,
      verbose=0,
      mode='auto',
      restore_best_weights=best_weights
  )

def model_checkpoint_callback(checkpoint_path, monitor='val_accuracy', save_best=True, weights=True):
  """
  Takes a path object to save the model to a folder. The monitor can be problem dependent,
  but is validation accuracy bu default. Will save the best validation accuracy by default. Will save only
  the model's weights instea of the entire model by default.
  """
  
  model_checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                      monitor=monitor,
                                                      save_best_only=True,
                                                      weights_only=weights,
                                                      verbose=0)
  return model_checkpoint

def compare_histories_fine_tuned_model(original_history, new_history, initial_epochs=5):
  """
  Compares two Tensorflow History objects, especially when fine-tuning a model. Metrics must be accuracy.

  """
  acc= original_history.history['accuracy']
  loss=original_history.history['loss']

  val_acc= original_history.history['val_accuracy']
  val_loss=original_history.history['val_loss']

  total_acc= acc+ new_history.history['accuracy']
  total_loss= loss+ new_history.history['loss']

  total_val_acc= val_acc+ new_history.history['val_accuracy']
  total_val_loss= val_loss+ new_history.history['val_loss']

  plt.figure(figsize=(8,8))
  plt.subplot(2,1,1)
  plt.plot(total_acc,label= 'Training Accuracy')
  plt.plot(total_val_acc, label='Val Accuracy')
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label= 'Start Fine Tuning' )
  plt.legend(loc='lower right')
  plt.title('Train and Validation Accuracy');

  plt.figure(figsize=(8,8))
  plt.subplot(2,1,2)
  plt.plot(total_loss, label= 'Training Loss')
  plt.plot(total_val_loss, label='Val Loss')
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label= 'Start Fine Tuning' )
  plt.legend(loc='upper right')
  plt.title('Train and Validation Loss');

def load_and_prep_image(filename, img_shape=[224,224], scale= True):
  """
  Reads an image from filename, turns it into a tensor and turns into
  img shape, img shape, color dim"""
  
  
  img= tf.io.read_file(filename)

  img= tf.image.decode_image(img)

  img= tf.image.resize(img, img_shape)

  if scale:
    img=img /255.

  return img

def pred_and_plot_binary(model, filename, class_names, img_shape=[224,224],  scale= True):
  """
  Imports an imaged located at filename, makes a prediction with model and plots
  the image with the predicted class as the title. Need to import Tensorflow, 
  matplotlib.image as mpimg """
  img= load_and_prep_image(filename, img_shape, scale)

  pred= model.predict(tf.expand_dims(img, axis=0))
  print('pred:', pred)
  print('round', tf.round(pred))
  pred_class= class_names[int(tf.round(pred))]

  plt.imshow(img)
  plt.title(f'Prediction: {pred_class}')
  plt.axis(False);

def pred_and_plot_multiclass(model, filename, class_names, img_shape=[224,224],  scale=True):
  """
  Imports an imaged located at filename, makes a prediction with model and plots
  the image with the predicted class as the title. Need to import Tensorflow, 
  matplotlib.image as mpimg """
  img= load_and_prep_image(filename, img_shape, scale)

  pred= model.predict(tf.expand_dims(img, axis=0))
  pred=pred.squeeze()
  pred_class= class_names[tf.math.argmax(pred)]

  plt.imshow(img)
  plt.title(f'Prediction: {pred_class}')
  plt.axis(False);

def confusion_matrix_classifier(y_true, y_pred, classes= None, figsize= (10,10), text_size=15):
  import itertools
  from sklearn.metrics import confusion_matrix

  confusion_matrix(y_true,  tf.round(y_pred))
  

  #Create the confusion matrix
  cm= confusion_matrix(y_true,tf.round(y_pred))
  cm_norm=cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
  n_classes= cm.shape[0]
  fig,ax= plt.subplots(figsize=figsize)
  cax= ax.matshow(cm, cmap= plt.cm.Blues)
  fig.colorbar(cax)

  

  if classes:
    labels= classes
  else:
    labels=np.arange(cm.shape[0])

  ax.set(title= 'Confusion Matrix',
    xlabel= 'Predicted Label',
    ylabel='True Label',
    xticks=np.arange(n_classes),
    yticks=np.arange(n_classes),
    xticklabels=labels,
    yticklabels= labels )

  ax.xaxis.set_label_position('bottom')
  ax.xaxis.tick_bottom()
  
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize= text_size)

  ax.yaxis.label.set_size(text_size)
  ax.xaxis.label.set_size(text_size)
  ax.title.set_size(text_size)

  threshold= (cm.max()+cm.min())/2.

  #plot the text
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i,f'{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)',
            horizontalalignment= 'center',
            color= 'white' if  cm[i,j]> threshold else 'black',
            size=text_size)

def plot_random_image(model, images, true_labels, classes):
  import random
  
  """
  Picks a random image, plots it and labels it with a prediction and truth label.
  """
  # Set up random integer
  i = random.randint(0, len(images))

  # Create predictions and targets
  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1, 28, 28))
  pred_label = classes[pred_probs.argmax()]
  true_label = classes[true_labels[i]]

  # Plot the image
  plt.imshow(target_image, cmap=plt.cm.binary)

  # Change the color of the titles depending on if the prediction is right or wrong
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"
  
  # Add xlabel information (prediction/true label)
  plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                   100*tf.reduce_max(pred_probs),
                                                   true_label),
             color=color) # set the color to green or red based on if prediction is right or wrong


def return_mae_mse(y_test, y_pred):
  mae=tf.metrics.mean_absolute_error(y_test,y_pred.squeeze())
  mse=tf.metrics.mean_squared_error(y_test,y_pred.squeeze())
  print(f'MAE: {mae}')
  print(f'MSE: {mse}')
  return(mae,mse)

def normalize(tensor):
  import numpy as np
  a=tensor
  b=np.linalg.norm(a)
  normalized_tensor=a/b
  return normalized_tensor

def create_model(model_url, num_classes=10):
  """
  Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in the output layer,
      should be equal to number of target classes, default 10.
  
  Returns:
    An uncompiled Keras Sequential model with model_url as feature extractor
    layer and Dense output layer with num_classes output neurons.
  """
  # Download the pretrained model and save it as a Keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False, # freeze the already learned patterns
                                           name="feature_extraction_lyaer",
                                           input_shape=IMAGE_SHAPE+(3,)) 

  # Create our own model
  model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(num_classes, activation="softmax", name="output_layer")
  ])
    
  return model  

def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

def plot_loss_curves(history):
  """
  Returns seperate loss curves for training and validation metrics using a history object from a tensorflow .fit function
  Note:Only viable when the model uses 'accuracy' as the model's metrics.
  import import matplotlib.pyplot as plt
  """
  loss = history.history[ 'loss']
  val_loss= history.history['val_loss']
  accuracy= history.history['accuracy']
  val_accuracy= history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  plt.plot(epochs, loss, label= 'training_loss' )
  plt.plot(epochs, val_loss, label= 'val_loss')
  plt.title("Loss")
  plt.xlabel('Epochs')
  plt.legend()
  
  plt.figure()
  plt.plot(epochs, accuracy, label= 'accuracy' )
  plt.plot(epochs, val_accuracy, label= 'val_accuracy')
  plt.title("Accuracy")
  plt.xlabel('Epochs')
  plt.legend()

def unzip_data(pathname):
  """
  Unzips a .zip file
  """
  
  zip_ref=zipfile.ZipFile(pathname)
  zip_ref.extractall()
  zip_ref.close()
  
def preprocess_img_dtype_resize_rescale(image, label, datatype=tf.float32,  img_shape=224, scale=False):
  """
  Converts image datatype to dytpe and reshapes image to 
  [img_shape, img_shape, color_channels]

  Needs tensorflow imported.
  """
  image=tf.image.resize(image, [img_shape,img_shape])
  if scale:
    image=image/255.
    return tf.cast(image, datatype), label
  else:
    return tf.cast(image, datatype), label 
  
def walk_through_dir(directory):
  for dirpath, dirnames, filenames in os.walk(directory):
    print(f'There are {len(dirnames)} directories and {len(filenames)} images in "{dirpath}" .')
