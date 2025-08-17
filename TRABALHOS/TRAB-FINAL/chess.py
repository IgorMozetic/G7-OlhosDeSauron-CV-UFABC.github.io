# GRUPO 07 - Olho de Sauron
# Igor Domingos da Silva Mozetic    - 11202320802
# Jhonattan Ferreira Machado        - 11202320245
# Mikael Alves Monteiro             - 21055813
# Script para Captura de Imagens de Calibração
# Data: 11/08/2025
# Nome do Arquivo: chess.py 
# Prompt do Linux: python3 chess.py

import cv2 as cv
import os

# --- Configurações ---
NOME_IMAGEM = "imgs"  
CAM_INDEX = 0            
SAVE_PATH = "images_webcam_chess"

# Cria a pasta de salvamento se ela não existir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Inicializa a captura de vídeo
cap = cv.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a câmera com índice {CAM_INDEX}")
    exit()

img_counter = 0
print("Pressione a tecla ESPACO para capturar uma imagem.")
print("Pressione a tecla ESC para sair.")

while True:
    # Captura quadro a quadro
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível receber o quadro da câmera. Saindo...")
        break

    cv.imshow('GRUPO 07 - Olho de Sauron - Captura de Calibracao', frame)
    k = cv.waitKey(1)

    if k % 256 == 27:  # Tecla ESC
        print("Saindo do programa.")
        break
    elif k % 256 == 32:  # Tecla ESPAÇO
        img_name = f"{SAVE_PATH}/{NOME_IMAGEM}_calib_{img_counter}.png"
        cv.imwrite(img_name, frame)
        print(f"Imagem salva: {img_name}")
        img_counter += 1

# Libera a captura e fecha todas as janelas
cap.release()
cv.destroyAllWindows()
          
