
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

#############
# Libraries #
#############

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from glob import glob
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

def log_spectrogram(audio, sample_rate = 16000, window_size = 20, step_size = 10, eps = 1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = spectrogram(audio, fs = sample_rate, window = "hann", nperseg = nperseg, noverlap = noverlap, detrend = False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def get_labels_filenames(dirpath, extension = "wav"):
    filepaths = glob(dirpath + r"*/*" + extension)
    
    pattern = r".+/(\w+)/\w+\." + extension + "$" # regex the label
    labels = []
    for filepath in filepaths:
        label = re.match(pattern, filepath)
        if label:
            labels.append(label.group(1))
            
    pattern = r".+/(\w+\." + extension + ")$" # regex the filename
    filenames = []
    for filepath in filepaths:
        filename = re.match(pattern, filepath)
        if filename:
            filenames.append(filename.group(1))
            
    return labels, filenames

def pad_audio(samples): # pad audios less than 16000 (1 second) with 0s
    if len(samples) >= sample_rate:       
        return samples
    else:
        return np.pad(samples, pad_width = (sample_rate - len(samples), 0), mode = "constant", constant_values = (0, 0))

def chop_audio(samples, sample_rate = 16000, num = 10000): # chop audios larger than 16000 (background noise wavs) to 16000 in length and create chunks out of one large wav files
    for i in range(num):
        beg = np.random.randint(0, len(samples) - sample_rate)
        yield samples[beg: beg + sample_rate]

def encode_labels(labels): # transform labels into dummies
    new_labels = []
    
    for label in labels:
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

labels_train, filenames_train = get_labels_filenames(train_path)

y_train = []
x_train = []

for label, filename in zip(labels_train, filenames_train):
    sample_rate, samples = wavfile.read(train_path + "/" + label + "/" + filename)
    samples = pad_audio(samples)
    
    if len(samples) > 16000:
        n_samples = chop_audio(samples)
    else:
        n_samples = [samples]
        
    for samples in n_samples:
        _, _, specgram = log_spectrogram(samples)
        y_train.append(label)
        x_train.append(specgram)
        
x_train = np.array(x_train)
x_train = np.expand_dims(x_train, axis = 3)

y_train = encode_labels(y_train) 
label_index = y_train.columns.values # used by pandas to create dummy values, we need it for later use.
y_train = np.array(y_train.values)

#################
# Visualization #
#################

plt.figure(figsize = (10, 10))
for i in range(9, 18):
    plt.subplot(3, 3, i - 9 + 1)
    plt.title(labels_train[i])    
    plt.imshow(x_train[i].squeeze().T, aspect = "auto", origin = "lower")
    plt.axis("off")

#################
# Specify model #
#################

inp = Input(shape = (99, 161, 1))

x = BatchNormalization()(inp)
x = Conv2D(8, kernel_size = (2, 2), activation = "relu")(x)
x = Conv2D(8, kernel_size = (2, 2), activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(rate = 0.2)(x)
x = Conv2D(16, kernel_size = (3, 3), activation = "relu")(x) # (3, 3)
x = Conv2D(16, kernel_size = (3, 3), activation = "relu")(x) # (3, 3)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(rate = 0.2)(x)
x = Conv2D(32, kernel_size = (3, 3), activation = "relu")(x)
x = Conv2D(32, kernel_size = (3, 3), activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(rate = 0.2)(x)
x = Conv2D(64, kernel_size = (3, 3), activation = "relu")(x)
x = Conv2D(64, kernel_size = (3, 3), activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(rate = 0.5)(x)
x = Dense(256, activation = "relu")(x)
x = BatchNormalization()(x)
outp = Dense(12, activation = "softmax")(x)

model = Model(inp, outp)

model.compile(
        loss = "binary_crossentropy",
        optimizer = Adam(lr = 1e-3),
#        optimizer = SGD(lr = 0.01, decay = 1e-4, momentum = 0.9, nesterov = True),
        metrics = ["accuracy"]
)

model.summary()

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
        batch_size = 64,
        validation_data = (x_val, y_val),
        epochs  = 9,
        shuffle = True,
        callbacks = callbacks_list
)

model.save("cnn.model")

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
        
        if i == batch_size: # stop at batch size
            i = 0
            imgs = np.array(imgs)
            imgs = np.expand_dims(imgs, axis = 3)
            yield filenames, imgs
            
    if i < batch_size: # leftover batch
        imgs = np.array(imgs)
        imgs = np.expand_dims(imgs, axis = 3)
        yield filenames, imgs
        
    raise StopIteration()
    
del x_train, y_train

filenames = []
labels = []

for name, img in test_gen(batch_size = 64):
    pred = model.predict(img)
    pred = np.argmax(pred, axis = 1)
    pred = list(label_index[i] for i in pred)
    filenames.extend(name)
    labels.extend(pred)

test_df = pd.DataFrame({"fname": filenames, "label": labels})
test_df.to_csv("submission.csv", index = False)
