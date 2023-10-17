import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils as np_utils
import pandas as pd
from collections import Counter
# from pyriemann.utils.viz import plot_confusion_matrix
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def check_folders():
  if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
  if not os.path.exists("results_opt"): os.makedirs("results_opt")
  data_f = "data"
  filename = "data_preprocessed_python.zip"
  url_data = "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html"
  # if not os.path.exists(data_f): os.makedirs(path)
  if not os.listdir(data_f):
    print("Request access to DEAP {0} at: {1}".format(filename, url_data))
    sys.exit()

def get_tag_AV(labels, type_class):
  _tags = []
  for l in labels:
    _arousal = l[0]
    _valence = l[1]
    p1 = ""
    if type_class=="Arousal":
      # p1 = "LA" if _arousal< 5 else "HA"
      p1 = 0 if _arousal< 5 else 1
    elif type_class=="Valence":
      # p1 = "LV" if _valence< 5 else "HV"
      p1 = 0 if _valence< 5 else 1
    _tags.append(p1)
  return _tags

def balance_dataset(matFeats, vecClass):
  mat_v = []
  vec_v = []
  _tags_c = Counter(vecClass)
  min_v = min(_tags_c.values())
  _ks = list(_tags_c.keys())
  counter_tags = [0]*len(_ks)
  for i, _c in enumerate(vecClass):
    _index = _ks.index(_c)
    if counter_tags[_index]<min_v:
      counter_tags[_index]+=1
      mat_v.append(matFeats[i])
      vec_v.append(_c)
  return mat_v, vec_v

def get_dataset(_step, subject, type_class):
  sample_rate = 128
  s_step = sample_rate*_step
  matFeats = []
  vecClass = []
  dataset = pickle.load(open('data/s{:02d}.dat'.format(subject), 'rb'), encoding='iso-8859-1')
  data = dataset['data']
  labels = dataset['labels']
  tags  = get_tag_AV(labels, type_class)
  for i, trial in enumerate(data):
    _start = sample_rate*3 # Removing the first 3 sec
    _end =  _start+s_step
    for segment_trial in range(int(60/_step)):
      _segment = trial[:32,_start:_end]
      matFeats.append(_segment)
      vecClass.append(tags[i])
      _start =_end
      _end = _start+s_step
  matFeats, vecClass = balance_dataset(matFeats, vecClass)
  print("+*"*100)
  print(Counter(vecClass))
  print("+*"*100)
  return {"data": np.array(matFeats), "tags":np.array(vecClass)}

def sub_dataset(dataset, chromosome):
  matFeats = []
  for trial in dataset["data"]:
      _segment = np.array([trial[k] for k,ch in enumerate(chromosome) if ch==1])
      matFeats.append(_segment)
  return np.array(matFeats), dataset["tags"]

def test_training(matFeats, vecClass):
  X, X_test, y, y_test = train_test_split(
    matFeats, vecClass, test_size=0.5) # , random_state=21
  X_validate, X_test, y_validate, y_test = train_test_split(
    X_test, y_test, test_size=0.5) #, random_state=21
  X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
  X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2], 1)
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
  y = np_utils.to_categorical(y)
  y_validate = np_utils.to_categorical(y_validate)
  y_test = np_utils.to_categorical(y_test)
  return X, X_validate, X_test, y, y_validate, y_test

def train_model(subject, s_ran, X, X_validate, X_test, y, y_validate, y_test, epochs_fit):
  no_instances, no_chs, no_datapoints, _kernel = X.shape
  model = EEGNet(nb_classes = 2, Chans = no_chs, Samples = no_datapoints, 
    dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, dropoutType = 'Dropout')
  model.compile(loss='categorical_crossentropy', optimizer='adam',
    metrics = ['accuracy'])
  # count number of parameters in the model
  numParams    = model.count_params()
  checkpointer = ModelCheckpoint(filepath='checkpoints/checkpoint_5s{0}{1}.h5'.format(subject, s_ran), verbose=1,
    save_best_only=True)
  class_weights = {0:1, 1:1}
  ################################################################################
  fittedModel = model.fit(X, y, batch_size = 16, epochs =epochs_fit,
    verbose = 2, validation_data=(X_validate, y_validate),
    callbacks=[checkpointer], class_weight = class_weights)
  # load optimal weights
  model.load_weights('checkpoints/checkpoint_5s{0}{1}.h5'.format(subject, s_ran))
  probs       = model.predict(X_test)
  preds       = probs.argmax(axis = -1)  
  acc         = np.mean(preds == y_test.argmax(axis=-1))
  print("Classification accuracy: %f " % (acc))
  # names = ["LA", "HA"]
  # plot_confusion_matrix(preds, y_test.argmax(axis = -1), names, title = 'EEGNet, Emotion classification')
  return acc

