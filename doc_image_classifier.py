import keras
import utils
import numpy as np
import os


BATCH_SIZE = 100
EPOCHS = 10
INITIAL_LR = 0.05
MODEL_PATH = 'doc_img_classifier.h5'


def lr_rate(epoch):

    return INITIAL_LR*(0.8 ** epoch)


class DocImageClassifier:

    def __init__(self, path=None):
        if path is None:
            self.model = self.build_model()
        else:
            self.model = keras.models.load_model(path)

    @staticmethod
    def build_model():

        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(None, None, 1)))
        model.add(keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
        model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
        model.add(keras.layers.GlobalAveragePooling2D())
        #model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(keras.optimizers.SGD(lr=INITIAL_LR, momentum=0.9),
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
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
        predictions = self.predict(paths)
        res = (predictions == labels)
        for i in range(res.shape[0]):
            if res[i] == 0.0:
                print(paths[i])
        return np.mean(predictions == labels)

    def predict(self, paths):
        predictions = []
        print(len(paths))
        for path in paths:
            X = utils.get_images_from_path(path)
            y = self.model.predict(X, verbose=0)
            predictions.append(np.round(np.mean(y)))
        return np.asarray(predictions)


if __name__ == '__main__':

    X_doc_train, X_doc_test, X_doc_val = utils.split_train_test_val(utils.DOC_PATH, 0.7, 0.15)
    X_img_train, X_img_test, X_img_val = utils.split_train_test_val(utils.IMG_PATH, 0.7, 0.15)

    X_train = X_doc_train + X_img_train
    X_test = X_doc_test + X_img_test
    X_val = X_doc_val + X_img_val

    y_train = np.ones(len(X_train)) * utils.IMAGE
    y_train[:len(X_doc_train)] = utils.DOCUMENT

    y_test = np.ones(len(X_test)) * utils.IMAGE
    y_test[:len(X_doc_test)] = utils.DOCUMENT

    y_val = np.ones(len(X_val)) * utils.IMAGE
    y_val[:len(X_doc_val)] = utils.DOCUMENT

    classifier = DocImageClassifier(MODEL_PATH)
    #train_gen = utils.image_gen(X_train, y_train, True)
    #val_gen = utils.image_gen(X_val, y_val, True)
    #classifier.fit(train_gen, len(X_train), val_gen, len(X_val))

    print("Test accuracy on %d examples: %f" % (len(X_test), classifier.evaluate(X_test, y_test)))




