import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import SGD
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import cv2
from skimage import feature, transform

class_labels = [
    'HighDanger', 'LowDanger', 'MediumDanger', 'Normal'
]

train1_data = r'D:\LEVEL_AHAR\Train'
validation1_data = r'D:\LEVEL_AHAR\Test'
IMG_SIZE = 160
LR = 1e-3
num_classes = len(class_labels)

x5 = class_labels.copy()
print(class_labels)

input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Block 1
layer0 = Conv2D(32, (7, 7), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv1')(input)
layer0 = BatchNormalization(name='bn1')(layer0)
layer0 = Activation('relu', name='relu1')(layer0)
layer0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='mp1')(layer0)

# skip_connection_1 = layer0

# Block 2
layer1 = Conv2D(48, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv2')(layer0)
layer1 = BatchNormalization(name='bn2')(layer1)
layer1 = Activation('relu', name='relu2')(layer1)

layer2 = Conv2D(48, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv3')(layer1)
layer2 = BatchNormalization(name='bn3')(layer2)
layer2 = Activation('relu', name='relu3')(layer2)

layer3 = Conv2D(48, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv4')(layer2)
layer3 = BatchNormalization(name='bn4')(layer3)
layer3 = Activation('relu', name='relu4')(layer3)

layer4 = Conv2D(48, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv5')(layer3)
layer4 = BatchNormalization(name='bn5')(layer4)
layer4 = Activation('relu', name='relu5')(layer4)

# layer5 = keras.layers.concatenate([skip_connection_1, layer4])
layer5 = Conv2D(48, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv6')(layer5)
layer5 = BatchNormalization(name='bn6')(layer5)
layer5 = Activation('relu', name='relu6')(layer5)
layer5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='mp2')(layer5)

# skip_connection_2 = layer5

# Block 3
layer6 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv7')(layer5)
layer6 = BatchNormalization(name='bn7')(layer6)
layer6 = Activation('relu', name='relu7')(layer6)

layer7 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv8')(layer6)
layer7 = BatchNormalization(name='bn8')(layer7)
layer7 = Activation('relu', name='relu8')(layer7)

layer8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv9')(layer7)
layer8 = BatchNormalization(name='bn9')(layer8)
layer8 = Activation('relu', name='relu9')(layer8)

layer9 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv10')(layer8)
layer9 = BatchNormalization(name='bn10')(layer9)
layer9 = Activation('relu', name='relu10')(layer9)

# layer10 = keras.layers.concatenate([skip_connection_2, layer9])
layer10 = Conv2D(64, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv11')(layer10)
layer10 = BatchNormalization(name='bn11')(layer10)
layer10 = Activation('relu', name='relu11')(layer10)
layer10 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='mp3')(layer10)

# skip_connection_3 = layer10

# Output block
layer11 = Conv2D(num_classes, (1, 1), kernel_regularizer=keras.regularizers.l2(1e-4), name='sep_conv22')(layer10)
layer11 = Flatten(name='flatten1')(layer11)
output = Activation('softmax', name='softmax1')(layer11)



model = Model(inputs=[input], outputs=[output])
model.summary()


# Image data generator with augmentation and preprocessing steps
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    train1_data,
    target_size=(IMG_SIZE, IMG_SIZE),
    classes=x5,
    batch_size=16,
    class_mode='categorical',
    color_mode='grayscale',  # Set color_mode to 'grayscale' for single-channel images
    shuffle=True,
    subset='training',  # Specify that this is the training subset
    interpolation='bilinear',
    seed=42
)

test_set = test_datagen.flow_from_directory(
    validation1_data,
    target_size=(IMG_SIZE, IMG_SIZE),
    classes=x5,
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale',  # Set color_mode to 'grayscale' for single-channel images
    interpolation='bilinear',
    seed=42
)


# Model compilation with class weights
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#Train the model
model.fit(
    training_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_data=test_set,
    validation_steps=len(test_set),
)

# Evaluate the model on the test set
eval_result = model.evaluate(test_set, steps=len(test_set))

# Extract predictions and true labels
predictions = model.predict(test_set, steps=len(test_set))
true_labels = test_set.classes


# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
conf_matrix_percentage = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix (Percentage)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Binarize the true labels
true_labels_binarized = label_binarize(true_labels, classes=range(num_classes))

# Plot the ROC curve for each class and calculate the AUC score
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red']

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(true_labels_binarized[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{class_labels[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
