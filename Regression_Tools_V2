def load_and_prep_image(filename, img_shape=[224,224]):
  """
  Reads an image from filename, turns it into a tensor and turns into
  img shape, img shape, color dim"""
  
  
  img= tf.io.read_file(filename)

  img= tf.image.decode_image(img)

  img= tf.image.resize(img, img_shape)

  img=img /255.

  return img

def pred_and_plot_binary(model, filename,img_shape=[224,224], class_names= class_names):
  """
  Imports an imaged located at filename, makes a prediction with model and plots
  the image with the predicted class as the title. Need to import Tensorflow, 
  matplotlib.image as mpimg """
  img= load_and_prep_image(filename)

  pred= model.predict(tf.expand_dims(img, axis=0))
  print('pred:', pred)
  print('round', tf.round(pred))
  pred_class= class_names[int(tf.round(pred))]

  plt.imshow(img)
  plt.title(f'Prediction: {pred_class}')
  plt.axis(False);

def pred_and_plot_multiclass(model, filename,img_shape=[224,224], class_names= class_names):
  """
  Imports an imaged located at filename, makes a prediction with model and plots
  the image with the predicted class as the title. Need to import Tensorflow, 
  matplotlib.image as mpimg """
  img= load_and_prep_image(filename)

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


def return_mae_mse(y_test=y_test, y_pred=y_pred):
  mae=tf.metrics.mean_absolute_error(y_test,y_pred.squeeze())
  mse=tf.metrics.mean_squared_error(y_test,y_pred.squeeze())
  print(f'MAE: {mae}')
  print(f'MSE: {mse}')
  return(mae,mse)

def normalize(tensor)
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

def walk_through_dir(directory):
  for dirpath, dirnames, filenames in os.walk(directory):
    print(f'There are {len(dirnames)} directories and {len(filenames)} images in "{dirpath}" .')
