import os
import cv2
import numpy as np  # Você esqueceu de importar o numpy

def preProcess(img):
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    imgPre = cv2.Canny(imgPre, 90, 140)
    kernel = np.ones((4, 4), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=2)
    imgPre = cv2.erode(imgPre, kernel, iterations=1)
    return imgPre

def captureImages(coinName: str, cleanFolder: bool = False) -> None:
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        return

    indexImage = 1

    while True:
        ret, image = video.read()
        if not ret:
            print("Erro: Não foi possível capturar a imagem da câmera")
            break

        image = cv2.resize(image, (640, 480))
        imagePreProcessed = preProcess(image)
        contours, _ = cv2.findContours(imagePreProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                imageCropped = image[y:y + h, x:x + w]
                imageCropped = cv2.resize(imageCropped, (224, 224))
                cv2.imshow('Image', imageCropped)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    os.makedirs('teste', exist_ok=True)  # Garante que a pasta exista
                    cv2.imwrite(f'teste/imagem{indexImage}.jpg', imageCropped)
                    print(f'Saving image number: {indexImage}')
                    indexImage += 1
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    video.release()
                    return

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    captureImages('imagens_treino')

