import os
import cv2
from src.PreProcessing import *

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
        imagePreProcessed = ImagePreProcessing(image)
        countors, _ = cv2.findContours(imagePreProcessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in countors:
            area = cv2.contourArea(cnt)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                imageCropped = image[y:y + h, x:x + w]
                imageCropped = cv2.resize(imageCropped, (224, 224))
                cv2.imshow('Image', imageCropped)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
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
    captureImages('teste')

