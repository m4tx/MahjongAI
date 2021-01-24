from tensorflow.keras.utils import to_categorical

from tenhou_log_utils.data_loader import load_data

train_x_data_discard, train_y_data_discard, train_x_data_pon, train_y_data_pon, train_x_data_chii, train_y_data_chii, train_x_data_riichi, train_y_data_riichi = load_data('train/01')

test_x_data_discard, test_y_data_discard, test_x_data_pon, test_y_data_pon, test_x_data_chii, test_y_data_chii, test_x_data_riichi, test_y_data_riichi = load_data('test')

epochs = 5

from .net import *

history = discard_model.fit(
    train_x_data_discard,
    to_categorical(train_y_data_discard),
    epochs=epochs,
    validation_data=(test_x_data_discard, to_categorical(test_y_data_discard)),
)
history = pon_model.fit(
    train_x_data_pon,
    to_categorical(train_y_data_pon),
    epochs=epochs,
    validation_data=(test_x_data_pon, to_categorical(test_y_data_pon)),
)
history = chii_model.fit(
    train_x_data_chii,
    to_categorical(train_y_data_chii),
    epochs=epochs,
    validation_data=(test_x_data_chii, to_categorical(test_y_data_chii)),
)
history = riichi_model.fit(
    train_x_data_riichi,
    to_categorical(train_y_data_riichi),
    epochs=epochs,
    validation_data=(test_x_data_riichi, to_categorical(test_y_data_riichi)),
)
#
# print(history.history)
#
discard_model.save_weights('discard.h5')
pon_model.save_weights('pon.h5')
chii_model.save_weights('chii.h5')
riichi_model.save_weights('riichi.h5')
