from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Flatten,
                                     BatchNormalization, Dropout)

num_filters = 100
filter_size = (5, 2)
dropout_rate = 0.5

discard_model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(43, 34, 4), padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Conv2D(num_filters, filter_size, padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Conv2D(num_filters, filter_size, padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Flatten(),
    Dense(300, activation='relu'),
    Dense(34, activation='relu'),
])

pon_model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(43, 34, 4), padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Conv2D(num_filters, filter_size, padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Conv2D(num_filters, filter_size, padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Flatten(),
    Dense(300, activation='relu'),
    Dense(2, activation='relu'),
])

chii_model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(43, 34, 4), padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Conv2D(num_filters, filter_size, padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Conv2D(num_filters, filter_size, padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Flatten(),
    Dense(300, activation='relu'),
    Dense(4, activation='relu'),
])

riichi_model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(43, 34, 4), padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Conv2D(num_filters, filter_size, padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Conv2D(num_filters, filter_size, padding='same'),
    BatchNormalization(),
    Dropout(rate=dropout_rate),
    Flatten(),
    Dense(300, activation='relu'),
    Dense(2, activation='relu'),
])

discard_model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

pon_model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

chii_model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

riichi_model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

discard_model.load_weights('discard.h5')
pon_model.load_weights('pon.h5')
chii_model.load_weights('chii.h5')
riichi_model.load_weights('riichi.h5')
