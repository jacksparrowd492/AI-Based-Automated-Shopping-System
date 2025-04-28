import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Dataset path
train_dir = r"D:\Projects\DL\Grocery_app\Model\Biscuit Wrappers Dataset"

# Image settings
img_height = 224
img_width = 224
batch_size = 32

# Data augmentation settings
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

# Validation data generator
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Check image shape
images, labels = next(train_generator)
print("Shape of first batch of images:", images.shape)

# Number of classes
num_classes = len(train_generator.class_indices)

# Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Initially freeze all layers
base_model.trainable = False

# Build model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, min_lr=1e-7)
checkpoint = ModelCheckpoint(r"D:\Projects\DL\Grocery_app\Model\Best_Biscuit_Model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# First Training Phase
epochs = 30
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, lr_scheduler, checkpoint]
)

# --- Fine-tune Entire VGG16 ---
print("🔓 Unfreezing all layers for fine-tuning...")
base_model.trainable = True

# Compile again with lower learning rate
fine_tune_lr = 1e-5
model.compile(optimizer=Adam(learning_rate=fine_tune_lr),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# Fine-tuning Phase
fine_tune_epochs = 20
total_epochs = epochs + fine_tune_epochs

history_finetune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    callbacks=[early_stopping, lr_scheduler, checkpoint]
)

# Save final fine-tuned model
model.save(r"D:\Projects\DL\Grocery_app\Model\Final_Biscuit_Model.h5")

print("✅ Biscuit Wrapper Model trained, fine-tuned, and saved successfully!")
