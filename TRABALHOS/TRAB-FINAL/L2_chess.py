# L2_chess.py - Script para Captura de Imagens de Calibração
import cv2 as cv
import os

# --- Configurações ---
NOME_ALUNO = "seu_nome"  # Altere para o nome de um integrante da equipe
CAM_INDEX = 0            # Índice da câmera (0 geralmente é a webcam interna)
SAVE_PATH = "images_webcam" # Pasta para salvar as imagens

# Cria a pasta de salvamento se ela não existir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Inicializa a captura de vídeo
cap = cv.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a câmera com índice {CAM_INDEX}")
    exit()

img_counter = 0
print("Pressione a tecla ESPAcO para capturar uma imagem.")
print("Pressione a tecla ESC para sair.")

while True:
    # Captura quadro a quadro
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível receber o quadro da câmera. Saindo...")
        break

    # Exibe o quadro resultante
    cv.imshow('Captura de Calibracao - Pressione ESPACO para salvar, ESC para sair', frame)

    # Aguarda por uma tecla
    k = cv.waitKey(1)

    if k % 256 == 27:  # Tecla ESC
        print("Saindo do programa.")
        break
    elif k % 256 == 32:  # Tecla ESPAÇO
        img_name = f"{SAVE_PATH}/{NOME_ALUNO}_calib_{img_counter}.png"
        cv.imwrite(img_name, frame)
        print(f"Imagem salva: {img_name}")
        img_counter += 1

# Libera a captura e fecha todas as janelas
cap.release()
cv.destroyAllWindows()
          
