# L2_cal.py - Script de Calibração de Câmera
import numpy as np
import cv2 as cv
import glob

# --- 1. Definição dos Parâmetros Iniciais ---

# Dimensões do tabuleiro: número de cantos internos em largura e altura.
# Um tabuleiro com 9x7 quadrados tem 8x6 cantos internos.
CHECKERBOARD = (8, 6) 

# Critérios de término para o refinamento dos cantos.
# O algoritmo para quando a precisão (EPS) ou o número de iterações (MAX_ITER) é atingido.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- 2. Preparação dos Pontos de Objeto e Pontos de Imagem ---

# Vetores para armazenar os pontos 3D do mundo real e os pontos 2D da imagem.
objpoints = [] # Pontos 3D no espaço do mundo real
imgpoints = [] # Pontos 2D no plano da imagem

# Criação dos pontos de objeto 3D (coordenadas do mundo).
# São as coordenadas (X, Y, Z) dos cantos do tabuleiro.
# Assumimos que o tabuleiro está no plano Z=0.
# As coordenadas são em uma unidade arbitrária (ex: "unidades de quadrado").
# Ex: (0,0,0), (1,0,0), (2,0,0),..., (7,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# --- 3. Processamento das Imagens de Calibração ---

# Encontra todas as imagens .jpg no diretório especificado.
images = glob.glob('images_webcam/*.png') # Altere o caminho se necessário

for fname in images:
    img = cv.imread(fname)
    # Converte a imagem para escala de cinza, pois a detecção de cantos opera em monocromático.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Encontra os cantos do tabuleiro de xadrez.
    # A função retorna 'ret' (True se os cantos foram encontrados) e 'corners' (as coordenadas dos cantos).
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    # Se os cantos forem encontrados com sucesso:
    if ret == True:
        # Adiciona os pontos de objeto (são os mesmos para todas as imagens).
        objpoints.append(objp)

        # Refina a localização dos cantos para precisão sub-pixel.
        # Isso é crucial para uma boa calibração.
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Opcional: Desenha os cantos na imagem para visualização.
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('Imagem com Cantos Detectados', img)
        cv.waitKey(500) # Espera 500ms

cv.destroyAllWindows()

# --- 4. Calibração da Câmera ---

# A função calibrateCamera usa os pontos de objeto e de imagem para calcular os parâmetros.
# gray.shape[::-1] fornece a resolução da imagem (largura, altura).
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# --- 5. Apresentação dos Resultados ---

print("Calibração concluída com sucesso!")
print("\nMatriz da Câmera (K):")
print(mtx)
print("\nCoeficientes de Distorção (dist):")
print(dist)
print("\nVetores de Rotação (rvecs - um por imagem):")
# print(rvecs) # Descomente para ver todos
print(f"Total de {len(rvecs)} vetores de rotação.")
print("\nVetores de Translação (tvecs - um por imagem):")
# print(tvecs) # Descomente para ver todos
print(f"Total de {len(tvecs)} vetores de translação.")

# --- 6. Cálculo do Erro de Reprojeção ---

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"\nErro total médio de reprojeção: {mean_error / len(objpoints)}")
  
