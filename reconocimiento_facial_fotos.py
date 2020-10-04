from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from numpy import asarray, array, mean, std, sqrt, maximum, sum, square, concatenate, append
from cv2 import resize, imread, cvtColor, COLOR_BGR2RGB
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image


class Reconocimiento:

    def __init__(self, input_shape_=(160, 160, 3)):
        self._modelo = load_model('modelo/modelo/facenet_keras.h5')  # el input es (None, 160, 160, 3)
        self.input_shape_ = input_shape_
        self.detector = MTCNN()
        self.le = None
        self.clf = None

    @property
    def modelo(self):
        return self._modelo

    @modelo.setter
    def modelo(self, valor):
        print(f'Imposible modificar el modelo a {str(valor)}')

    @modelo.deleter
    def modelo(self):
        print('Imposible eliminar el modelo')

    def extraer_caras(self, rutas, p=0.1, escala=True):
        """
        Metodo para extraer las caras en fotografias almacenadas en una ruta en especifico
        :param ruta: path al archivo
        :param p: porcentaje de desfazaje mas alla del bounding box, util para aumentar el campo de visión
        :param escala: True implica que sera redimencionado a la medida del input shape, en False queda igual que el BB
        :return:
        """
        caras = []
        for archivos in rutas:
            aux = imread(archivos)
            im_rgb = cvtColor(aux, COLOR_BGR2RGB)
            boxes = self.detector.detect_faces(im_rgb)
            img_array = asarray(im_rgb)
            lista_caras = img_array[int((1 - 3 * p)*boxes[0]['box'][1]):int((1 + p) * (boxes[0]['box'][1] +
                                    boxes[0]['box'][3])),
                          int((1 - p)*boxes[0]['box'][0]):int((1 + p)*(boxes[0]['box'][0] + boxes[0]['box'][2])), :]
            resized = resize(lista_caras, dsize=(self.input_shape_[0], self.input_shape_[1]))
            caras.append(resized)
        return array(caras)


    @staticmethod
    def prewhiten(x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')
        mean_ = mean(x, axis=axis, keepdims=True)
        std_ = std(x, axis=axis, keepdims=True)
        std_adj = maximum(std_, 1.0 / sqrt(size))
        y = (x - mean_) / std_adj
        return y

    @staticmethod
    def l2_normalize(x, axis=-1, epsilon=1e-10):
        output = x/sqrt(maximum(sum(square(x), axis=axis, keepdims=True), epsilon))
        return output

    def embedings(self, filepaths, margin=12, batch_size=1):
        aligned_images = self.prewhiten(self.extraer_caras(filepaths))
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(self._modelo.predict_on_batch(aligned_images[start:start + batch_size]))
        embs = self.l2_normalize(concatenate(pd))
        return embs

    def train(self, dir_basepath, names, max_num_img=10):
        labels = []
        embs = []
        for name in names:
            dirpath = os.path.abspath(dir_basepath + name)
            filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
            embs_ = self.embedings(filepaths)
            labels.extend([name]*len(embs_))
            embs.append(embs_)
        embs = concatenate(embs)
        le = LabelEncoder().fit(labels)
        y = le.transform(labels)
        clf = SVC(kernel='linear', probability=True).fit(embs, y)
        self.le = le
        self.clf = clf

    def _infer(self, filepaths, recortar=True, umbral=0.5):
        if recortar:
            ca = self.extraer_caras(filepaths)
            tmp = Image.fromarray(ca[0])
            tmp.save('img_test/temporal.jpg')
            embs = self.embedings(['img_test/temporal.jpg'])
        else:
            embs = self.embedings(filepaths)
        pred_0 = self.clf.predict_proba(embs).max()
        print(pred_0*100, '% de seguridad')
        if pred_0 >= umbral:
            pred = self.le.inverse_transform(self.clf.predict(embs))
        else:
            pred = 'No se tiene suficiente certeza de quien sea la imagen'
        return pred

    def pred(self, filepath, recortar=False, umbral=0.5):
        if self.le and self.clf:
            prediccion = self._infer(filepath, recortar=recortar, umbral=umbral)
            return prediccion
        else:
            raise UserWarning("Primero debe entrenar el modelo con las imagenes")

    def __str__(self):
        return str(self._modelo.summary())


path_img = 'BD/'
nombres = os.listdir(path_img)
model = Reconocimiento()
print('Instancia creada')
model.train(path_img, nombres)
print('Modelo entrenado')
path_test = 'img_test'
print(model.pred(['img_test/2020-08-13-164807 (3.ª copia).jpg']))
predecir = True
while predecir:
    solicitud = input("Ingrese ruta a nuevo archivo o 'salir' para salir: ")
    if solicitud.lower() == 'salir':
        predecir = False
    else:
        try:
            prediccion_ = model.pred([solicitud], recortar=True, umbral=0.48)
            print(f'La persona en el archivo {solicitud} es:  {prediccion_}')
            plt.imshow(asarray(Image.open(solicitud)))
            plt.title(str(prediccion_[0]))
            plt.show()
        except Exception:
            pass
