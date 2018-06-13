import keras
import utils
import numpy as np
import os

BATCH_SIZE = 100
EPOCHS = 10
INITIAL_LR = 0.01
MODEL_PATH = '{}_{}_neighbors_classifier.h5'


def lr_rate(epoch):

    return INITIAL_LR*(0.8 ** epoch)


class SemiGlobalAvgPooling(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SemiGlobalAvgPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = (input_shape[0], input_shape[2])
        super(SemiGlobalAvgPooling, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return keras.backend.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3]


keras.layers.SemiGlobalAvgPooling = SemiGlobalAvgPooling


class NeighborsClassifier:

    def __init__(self, is_docs=True, is_left_right=True, load_model=False):
        self.is_left_right = is_left_right
        self.is_docs = is_docs
        self.model_path = MODEL_PATH.format('lr' if self.is_left_right else 'ud',
                                            'docs' if self.is_docs else 'imgs')

        if not load_model:
            self.model = self.build_model()
        else:
            self.model = keras.models.load_model(self.model_path)

    @staticmethod
    def build_model():

        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(8, (5, 5), padding='same', activation='relu', input_shape=(None, utils.STRIP_SIZE * 2, 1)))
        #model.add(keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'))
        #model.add(keras.layers.Conv2D(64, 5, activation='relu', padding='same'))
        model.add(SemiGlobalAvgPooling())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(keras.optimizers.SGD(lr=INITIAL_LR, momentum=0.9),
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
                      )

        model.summary()

        return model

    def fit(self, gen, steps, val_gen, val_steps):
        self.model.fit_generator(gen,
                                 steps_per_epoch=steps,
                                 epochs=EPOCHS,
                                 validation_data=val_gen,
                                 validation_steps=val_steps,
                                 workers=3,
                                 callbacks=[keras.callbacks.LearningRateScheduler(lr_rate)])

        self.model.save(self.model_path)

    def evaluate(self, paths):
        labels = np.asarray(labels)
        predictions = self.predict(paths)
        res = (predictions == labels)
        for i in range(res.shape[0]):
            if res[i] == 0.0:
                print(paths[i])
        return np.mean(predictions == labels)

    def predict(self, im1, im2):
        #im1 = utils.process_image(im1, None)
        #im2 = utils.process_image(im2, None)

        image = utils.create_images_strip(im1, im2, self.is_left_right, utils.STRIP_SIZE)
        image = np.expand_dims(image, 0)

        y = self.model.predict(image, verbose=0)
        return y


if __name__ == '__main__':

    #X_doc_train, X_doc_test, X_doc_val = utils.split_train_test_val(utils.DOC_PATH, 0.7, 0.15)
    X_img_train, X_img_test, X_img_val = utils.split_train_test_val(utils.IMG_PATH, 0.7, 0.15, seed=42)

    classifier = NeighborsClassifier(is_docs=False, is_left_right=True)
    train_gen = utils.neighbors_gen(X_img_train, utils.STRIP_SIZE, is_left_right=True)
    val_gen = utils.neighbors_gen(X_img_val, utils.STRIP_SIZE, is_left_right=True)
    classifier.fit(train_gen, len(X_img_train), val_gen, len(X_img_val))

    #print("Test accuracy on %d examples: %f" % (len(X_img_test), classifier.evaluate(X_img_test)))




