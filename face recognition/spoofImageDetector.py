# from keras import models , layers
import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
import splitfolders
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.vgg19 import VGG19

import joblib


# =============================================================================
#  image_dim = (180 , 180 )
# =============================================================================

class SpoofImageDetector:
    def __init__(self, epochs = 10  , version = 1,  train_path="faceClassification_v1/train", val_path="faceClassification_v1/val", path_model='',
                 learn='VGG19', image_dim=(224 , 224 )
                 , train=True):
        self.image_dim = image_dim

        if train:

            if learn == '':

                self.model = self.Model()
            elif learn == "VGG19":
                self.image_dim = (224, 224)
                self.model = self.transfertLearning(output=2)

            # self.model.summary()
            self.train_path = train_path
            self.model.compile(optimizer='adam',
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy'])
            train_x, train_y = self.getData(train_path)
            val_x, val_y = self.getData(val_path)

            self.history = self.model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs,
                                          batch_size=3)

            self.Evaluation()

            self.saveModele(f"model_{learn}_v{version}")

        else:
            self.model = keras.models.load_model(path_model)

    def Model(self , output = 2):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(output, activation="softmax"))

        return model

    def getData(self, path):
        x_train = []
        for folder in os.listdir(path):
            sub_path = path + "/" + folder
            for img in os.listdir(sub_path):
                image_path = sub_path + "/" + img
                img_arr = cv2.imread(image_path)
                img_arr = cv2.resize(img_arr, self.image_dim)
                x_train.append(img_arr)
        train_x = np.array(x_train)
        train_x = train_x / 255.0
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        training_set = train_datagen.flow_from_directory(path,
                                                         target_size=(180, 180),
                                                         batch_size=3,
                                                         class_mode='categorical')
        train_y = training_set.classes
        training_set.class_indices

        return train_x, train_y

    def spliteFolders(self, input_folder, output_folder):

        splitfolders.ratio(input_folder, output=output_folder,
                           seed=42, ratio=(.6, .2, .2), group_prefix=None)

    def transfertLearning(self , output = 3 ):
        vgg = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        # ne pas entrainer les couches pré-entrainées de VGG-19
        for layer in vgg.layers:
            layer.trainable = False

        x = keras.layers.Flatten()(vgg.output)

        # ajout d'une couche de sortie
        prediction = keras.layers.Dense(output, activation='softmax')(x)
        model = keras.models.Model(inputs=vgg.input, outputs=prediction)
        # model.summary()

        return model

    def saveModele(self, path_model):
        self.model.save(path_model, save_format='tf')

    def Evaluation(self):

        plt.plot(self.history.history['accuracy'], label='train acc')
        plt.plot(self.history.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()
        plt.plot(self.history.history['loss'], label='train loss')
        plt.plot(self.history.history['val_loss'], label='val loss')
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val-accuracy')
        plt.legend()
        plt.show()

    def Test(self, val_path, path_model):
        model = joblib.load(path_model)

        test_x, test_y = self.getData(val_path)
        model.evaluate(test_x, test_y, batch_size=32)
        y_pred = model.predict(test_x)
        y_pred = np.argmax(y_pred, axis=1)
        print(classification_report(y_pred, test_y))
        # print(confusion_matrix(y_pred,self.test_y))

    def predict_image(self, image_path):
        # Load the model from the model path

        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_dim)
        img = np.expand_dims(img, axis=0) / 255.0  # Normalization

        # Obtain the class prediction
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)

        return predicted_class[0]

    def test_with_frame(self, frame):
        # Preprocess the frame
        frame = cv2.resize(frame, self.image_dim)
        img = frame / 255.0  # Normalization
        img = np.expand_dims(img, axis=0)

        # Perform the prediction
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)

        return predicted_class[0]

    def test_with_video_cam(self):
        # Ouvrir la caméra vidéo
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        frame = cv2.resize(frame, self.image_dim)
        img = frame / 255.0  # Normalisation
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)

        while predicted_class[0] == 1:
            # Lire chaque frame de la caméra vidéo
            ret, frame = cap.read()

            # Redimensionner l'image
            frame = cv2.resize(frame, self.image_dim)

            # Prétraitement de l'image
            img = frame / 255.0  # Normalisation

            # Ajouter une dimension supplémentaire pour l'axe des lots
            img = np.expand_dims(img, axis=0)

            # Effectuer la prédiction
            prediction = self.model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)

            # Afficher le résultat de la prédiction sur l'image
            label = "Classe : {}".format(predicted_class[0])
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Afficher la frame avec la prédiction
            cv2.imshow('Video Prediction', frame)

            # Quitter si la touche 'q' est pressée
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libérer les ressources
        print(" etap 1 success :")
        cap.release()
        cv2.destroyAllWindows()


#cnn = SpoofImageDetector(epochs=10, version=6, train=True, path_model='model_VGG19_v5')

# cnn.Test("faceClassification_2/val","model_VGG19.pkl")

#cnn.spliteFolders("dataset_v1", "faceClassification_v1")

# =============================================================================
# Test Video
# =============================================================================

#cnn.test_with_video_cam()

# =============================================================================
# Test Image
# =============================================================================

"""
image_path = "img.jpg"
predicted_class = cnn.predict_image(image_path)
print(predicted_class)

"""

