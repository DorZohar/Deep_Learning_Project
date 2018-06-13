import keras
import utils
import numpy as np
import os


BATCH_SIZE = 1
EPOCHS = 10
INITIAL_LR = 0.1
MODEL_PATH = 'doc_img_classifier.h5'


def lr_rate(epoch):

    return INITIAL_LR*(0.9 ** epoch)


class TilePosRegressor:

    def __init__(self, load_model=False):
        if not load_model:
            self.model = self.build_model()
        else:
            self.model = keras.models.load_model(MODEL_PATH)

    @staticmethod
    def build_model():

        input_layer = keras.layers.Input(shape=(None, None, 1))
        conv1_layer = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(input_layer)
        conv2_layer = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(conv1_layer)
        conv3_layer = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv2_layer)
        pooling_layer = keras.layers.GlobalAveragePooling2D()(conv3_layer)
        dense = keras.layers.Dense(128, activation='relu')(pooling_layer)

        output_x = keras.layers.Dense(1, activation='sigmoid')(dense)
        output_y = keras.layers.Dense(1, activation='sigmoid')(dense)

        model = keras.models.Model(input_layer, [output_x, output_y])

        model.compile(keras.optimizers.SGD(lr=INITIAL_LR, momentum=0.9),
                      loss='mean_squared_error',
                      )

        model.summary()

        return model

    def fit(self, gen, steps, val_gen, val_steps):
        self.model.fit_generator(gen,
                                 steps_per_epoch=steps // BATCH_SIZE,
                                 epochs=EPOCHS,
                                 validation_data=val_gen,
                                 validation_steps=val_steps // BATCH_SIZE,
                                 callbacks=[keras.callbacks.LearningRateScheduler(lr_rate)])

        self.model.save(MODEL_PATH)

    def evaluate(self, paths, labels):
        labels = np.asarray(labels)
        predictions = self.predict_from_path(paths)
        res = (predictions == labels)
        for i in range(res.shape[0]):
            if res[i] == 0.0:
                print(paths[i])
        return np.mean(predictions == labels)

    def predict_from_path(self, paths):
        predictions = []
        print(len(paths))
        for path in paths:
            X = utils.get_images_from_path(path)
            y = self.predict(X)
            predictions.append(y)
        return np.asarray(predictions)

    def predict(self, images):
        y = self.model.predict(np.asarray(images), verbose=0)
        return np.round(np.mean(y))


if __name__ == '__main__':

    X_img_train, X_img_test, X_img_val = utils.split_train_test_val(utils.IMG_PATH, 0.7, 0.15, seed=42)

    classifier = TilePosRegressor()
    train_gen = utils.centers_gen(X_img_train)
    val_gen = utils.centers_gen(X_img_val)
    classifier.fit(train_gen, len(X_img_train), val_gen, len(X_img_val))




