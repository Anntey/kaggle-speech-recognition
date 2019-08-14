#############
# Libraries #
#############

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from re import match
from scipy.io import wavfile
from scipy.signal import spectrogram
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger
from keras.layers import Conv2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split

#########################
# Define some variables #
#########################

sample_rate = 16000

labels_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown"]

train_path = "./input/train/train/audio/"
test_path = "./input/test/test/audio/"

#####################
# Utility functions #
#####################

def log_spectrogram(audio, sample_rate = 16000, window_size = 20, step_size = 10, eps = 1e-10): # why transpose ??????????????????
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = spectrogram(audio, fs = sample_rate, window = "hann", nperseg = nperseg, noverlap = noverlap, detrend = False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def pad_audio(samples): # pad audios less than 16000 with 0s
    if len(samples) >= sample_rate:       
        return samples
    else:
        return np.pad(samples, pad_width = (sample_rate - len(samples), 0), mode = "constant", constant_values = (0, 0))

def chop_audio(samples, sample_rate = 16000, n = 10000): # chop audios larger than 16000 (background noise wavs) to 16000
    for i in range(n):
        beg = np.random.randint(0, len(samples) - sample_rate)
        yield samples[beg: beg + sample_rate]

def encode_labels(orig_labels): # transform labels into dummies
    new_labels = []
    
    for label in orig_labels:
        if label == "_background_noise_":
            new_labels.append("silence")
        elif label not in labels_list:
            new_labels.append("unknown")
        else:
            new_labels.append(label)
            
    return pd.get_dummies(pd.Series(new_labels))

#########################
# Prepare training data #
#########################

filepaths_train = glob(train_path + r"*/*" + "wav")
  
labels_train = []
pattern_label = r".+/(\w+)/\w+\." + extension + "$" # regex label

for filepath in filepaths_train:
    label = match(pattern_label, filepath)
    if label:
        labels_train.append(label.group(1)) # append capture group

filenames_train = []            
pattern_filename = r".+/(\w+\." + "wav" + ")$" # regex filename

for filepath in filepaths_train:
    filename = match(pattern_filename, filepath)
    if filename:
        filenames_train.append(filename.group(1)) # append capture group
            
y_train = []
x_train = []

for label, filename in zip(labels_train, filenames_train):
    _, sample = wavfile.read(train_path + label + "/" + filename) # "/" + sample_rate
    sample = pad_audio(samples)
    
    if len(sample) > 16000:
        n_samples = chop_audio(samples)
    else:
        n_samples = [sample] # list
        
    for sample in n_samples:
        _, _, specgram = log_spectrogram(sample)
        y_train.append(label)
        x_train.append(specgram)
        
x_train = np.array(x_train)
x_train = np.expand_dims(x_train, axis = 3) # reshape to (n, 99, 161, 1)

y_train = encode_labels(y_train) 
label_index = y_train.columns.values # used by pandas to create dummy values, we need it for later use.
y_train = np.array(y_train.values)

#################
# Visualization #
#################

import random
indices, locus = random.sample(range(0, len(x_train.shape[0])), 9), range(0, 9)

plt.figure(figsize = (10, 10))
for i, locus in zip(indices, locus):
    plt.subplot(3, 3, locus)
    plt.title(labels_train[i])    
    plt.imshow(x_train[i].squeeze().T, aspect = "auto", origin = "lower") 
    plt.axis("off")

#################
# Specify model #
#################

inp = Input(shape = (99, 161, 1))

x = BatchNormalization()(inp)
x = Conv2D(8, (2, 2), activation = "relu")(x)
x = Conv2D(8, (2, 2), activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(16, (3, 3), activation = "relu")(x)
x = Conv2D(16, (3, 3), activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(32, (3, 3), activation = "relu")(x)
x = Conv2D(32, (3, 3), activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation = "relu")(x)
x = Conv2D(64, (3, 3), activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation = "relu")(x)
x = BatchNormalization()(x)
outp = Dense(12, activation = "softmax")(x)

model = Model(inp, outp)

model.compile(
        loss = "binary_crossentropy", # categorical ????????????????
        optimizer = Adam(lr = 1e-3),
#       optimizer = SGD(lr = 1e-3, decay = 1e-4, momentum = 0.9, nesterov = True),
        metrics = ["accuracy"]
)

#############
# Fit model #
#############

x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size = 0.1,
        random_state = 2019
)

callbacks_list = [
        CSVLogger("training.log")
]

fit_log = model.fit(
        x_train,
        y_train,
        validation_data = (x_val, y_val),
        batch_size = 64,
        epochs  = 9,
        shuffle = True,
        callbacks = callbacks_list
)

del x_train, y_train

##############
# Evaluation #
##############

fit_log_df = pd.DataFrame(fit_log.history)
fit_log_df[["acc", "val_acc"]].plot()
fit_log_df[["loss", "val_loss"]].plot()

##############
# Prediction #
##############

def test_gen(batch_size):
    filepaths = glob(test_path + "*wav")
    i = 0
    
    for path in filepaths:
        
        if i == 0:
            imgs = []
            filenames = []
            
        i = i + 1        
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        _, _, specgram = log_spectrogram(samples)
        imgs.append(specgram)
        filenames.append(path.split("/")[-1]) # split path and get filename
        
        if i == batch_size: # continue until batch size
            i = 0
            imgs = np.array(imgs)
            imgs = np.expand_dims(imgs, axis = 3) #2???????
            yield filenames, imgs
            
    if i < batch_size: # leftover batch
        imgs = np.array(imgs)
        imgs = np.expand_dims(imgs, axis = 3)
        yield filenames, imgs
        
    raise StopIteration()

filenames = []
labels = []

for name, img in test_gen(batch_size = 64): # miten toimii???????
    pred = model.predict(img)
    pred = np.argmax(pred, axis = 1)
    pred = list(label_index[i] for i in pred)
    filenames.extend(name)
    labels.extend(pred)

test_df = pd.DataFrame({"fname": filenames, "label": labels})
test_df.to_csv("submission.csv", index = False)