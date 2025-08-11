import cv2
import numpy as np
from keras.models import load_model
import pickle

# --- Carrega os parâmetros de calibração da câmera ---
with open("calibracao_camera.pkl", "rb") as f:
    mtx, dist, _, _ = pickle.load(f)

# --- Inicializa a captura ---
video = cv2.VideoCapture(0)

model = load_model('modelo_keras/keras_model.h5', compile=False)
data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
classes = ["1real", "50cent", "10cent", "5cent"]
# classes = ["1real", "50cent", "25cent", "10cent", "5cent"]

def preProcess(img):
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    imgPre = cv2.Canny(imgPre,90,140)
    kernel = np.ones((4,4),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre

def DetectarMoeda(img):
    imgMoeda = cv2.resize(img,(224,224))
    imgMoeda = np.asarray(imgMoeda)
    imgMoedaNormalize = (imgMoeda.astype(np.float32)/127.0)-1
    data[0] = imgMoedaNormalize
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = classes[index]
    return classe,percent
    
while True:
    _, img = video.read()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]

    # Resize da imagem corrigida
    img = cv2.resize(img, (640, 480))

    imgPre = preProcess(img)
    countors, hi = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    qtd = 0
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            recorte = img[y:y + h, x:x + w]
            classe, conf = DetectarMoeda(recorte)
            if conf > 0.65:
                cv2.putText(img, str(classe), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if classe == '1real': qtd += 1
                if classe == '50cent': qtd += 0.5
                # if classe == '25cent': qtd += 0.25
                # if classe == '10cent': qtd += 0.10
                if classe == '5cent': qtd += 0.05

    cv2.rectangle(img, (430, 30), (600, 80), (0, 0, 255), -1)
    cv2.putText(img, f'R$ {qtd:.2f}', (440, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow('Imagem com as Moedas', img)
    cv2.imshow('Imagem Processada', imgPre)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

video.release()
cv2.destroyAllWindows()
