import numpy as np
import os
import csv
from PIL import Image

import keras
import keras.preprocessing.image as img
from keras.applications import ResNet50, VGG16
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.layers.pooling import GlobalMaxPool2D, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import backend as K

import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline

# get all the filenames
all_files = []
for path, subdirs, files in os.walk('data'):
    for name in files:
        all_files.append(os.path.join(path, name))

# crop the images and save in data_crop folder
for f in all_files:
    temp_img = Image.open(f)
    temp_img = temp_img.crop((0, 600 - 435, 800, 600 - 435 + 185))
    temp_img.save('data_crop' + f.split('data')[1])

train_files = []
for path, subdirs, files in os.walk('data/train/'):
    for name in files:
        train_files.append(os.path.join(path, name))

np.random.shuffle(train_files)

valid_files = train_files[:500]
train_files = train_files[500:]

for f in valid_files:
    os.rename(f, 'data/valid/' + f.split('data/train/')[1])


def imagenet_mean(x):
    x = x[..., ::-1]
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


train_gen = img.ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.05,
    height_shift_range=0.1,
    preprocessing_function=imagenet_mean
)
test_gen = img.ImageDataGenerator(
    preprocessing_function=imagenet_mean
)

batch_size=64
img_size = (300,300)


train_batches = train_gen.flow_from_directory(
    'data/train/',
    batch_size=batch_size,
    target_size = img_size,
    class_mode='binary'
)

valid_batches = test_gen.flow_from_directory(
    'data/valid/',
    batch_size=batch_size,
    target_size = img_size,
    shuffle=False,
    class_mode='binary'
)

test_batches = test_gen.flow_from_directory(
    'data/test/',
    batch_size=batch_size,
    target_size = img_size,
    shuffle=False,
    class_mode='binary'
)

temp_train_batch = train_batches.next()
print('X shape: ', temp_train_batch[0].shape)
print('Y shape: ', temp_train_batch[1].shape)

plt.imshow(temp_train_batch[0][0].astype('uint8'))


# choose the convnet
base_model = ResNet50(include_top=False, input_shape=img_size + (3,))
#base_model = densenet121_model(img_rows=img_size[0], img_cols=img_size[1], color_type=3, num_classes=2)
#base_model = resnet101_model(img_rows=img_size[0], img_cols=img_size[1], color_type=3, num_classes=2)

base_model.summary()

ft_map = base_model.get_layer(index=-2).output

x = Conv2D(128, (3,3), padding='same')(ft_map)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
x = GlobalAveragePooling2D()(x)

model = Model(base_model.input, x)

model.summary()

# freeze all the base model layers
for layer in base_model.layers:
    layer.trainable = False

opt = Adam(0.001)#, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches,
                    steps_per_epoch=np.ceil(train_batches.samples/batch_size),
                    epochs=5, verbose=1,
                    validation_data=valid_batches,
                    validation_steps=np.ceil(valid_batches.samples/batch_size),
                    )

model.save_weights('models/rn50_cls.h5')
K.set_value(model.optimizer.lr, 0.00001)
model.fit_generator(train_batches,
                    steps_per_epoch=np.ceil(train_batches.samples/batch_size),
                    epochs=5, verbose=1,
                    validation_data=valid_batches,
                    validation_steps=np.ceil(valid_batches.samples/batch_size),
                    )
model.save_weights('models/rn50_cls.h5')
for i,layer in enumerate(model.layers):
    print(i, layer.name)

for layer in model.layers[:141]:
    layer.trainable = False

for layer in model.layers[141:]:
    layer.trainable = True

opt = Adam(0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches,
                    steps_per_epoch=np.ceil(train_batches.samples/batch_size),
                    epochs=5, verbose=1,
                    validation_data=valid_batches,
                    validation_steps=np.ceil(valid_batches.samples/batch_size),
                    )
model.save_weights('models/rn50_block5.h5')
K.set_value(model.optimizer.lr, 0.00001)
model.fit_generator(train_batches,
                    steps_per_epoch=np.ceil(train_batches.samples/batch_size),
                    epochs=3, verbose=1,
                    validation_data=valid_batches,
                    validation_steps=np.ceil(valid_batches.samples/batch_size),
                    )
model.save_weights('models/rn50_block5.h5')
K.set_value(model.optimizer.lr, 0.000001)
model.fit_generator(train_batches,
                    steps_per_epoch=np.ceil(train_batches.samples/batch_size),
                    epochs=2, verbose=1,
                    validation_data=valid_batches,
                    validation_steps=np.ceil(valid_batches.samples/batch_size),
                    )

# load data in memory
valid_batches.reset()
x_valid = np.vstack([valid_batches.next()[0] for x in range(int(np.ceil(valid_batches.samples/batch_size)))])

valid_batches.reset()
y_valid = np.concatenate([valid_batches.next()[1] for x in range(int(np.ceil(valid_batches.samples/batch_size)))])

p_valid = np.zeros_like(y_valid)
for flip in [False, True]:
    temp_x = x_valid
    if flip:
        temp_x = img.flip_axis(temp_x, axis=2)
    p_valid += 0.5 * np.reshape(model.predict(temp_x, verbose=1), y_valid.shape)

np.mean((p_valid > 0.5) == y_valid)

# load data in memory
test_batches.reset()
x_test = np.vstack([test_batches.next()[0] for x in range(int(np.ceil(test_batches.samples/batch_size)))])

test_batches.reset()
y_test = np.concatenate([test_batches.next()[1] for x in range(int(np.ceil(test_batches.samples/batch_size)))])
p_test = np.zeros_like(y_test)
for flip in [False, True]:
    temp_x = x_test
    if flip:
        temp_x = img.flip_axis(temp_x, axis=2)
    p_test += 0.5 * np.reshape(model.predict(temp_x, verbose=1), y_test.shape)

from sklearn.metrics import confusion_matrix
np.mean((p_test > 0.5) == y_test)

cam_extract = Model(base_model.input, model.get_layer(index=-3).output)
cam_valid = cam_extract.predict(x_valid, verbose=1)
valid_ind = np.random.randint(low=0,high=500)
valid_file = valid_batches.filenames[valid_ind]
print(valid_file)
valid_cam = cam_extract.predict(np.expand_dims(x_valid[valid_ind], 0))
np.max(valid_cam)
overlay = img.array_to_img(valid_cam[0]).resize((800,600), Image.BILINEAR).convert('RGB')
bg = img.load_img('data/valid/' + valid_file)#.resize((300,300))
Image.blend(alpha=0.5, im1=bg, im2=overlay)
test_ind = np.random.randint(high=1500,low=0)
test_file = test_batches.filenames[test_ind]
print(test_file)
test_cam = cam_extract.predict(np.expand_dims(x_test[test_ind], 0))
np.max(test_cam)
overlay = img.array_to_img(test_cam[0]).resize((800,600), Image.BILINEAR).convert('RGB')
bg = img.load_img('data/test/' + test_file)#.resize((300,300))
Image.blend(alpha=0.5, im1=bg, im2=overlay)