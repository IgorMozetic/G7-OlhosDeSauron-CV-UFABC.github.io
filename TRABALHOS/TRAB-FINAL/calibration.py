# L2_cal.py - Script de Calibração de Câmera
import numpy as np
import cv2 as cv
import glob
import pickle

# --- 1. Definição dos Parâmetros Iniciais ---

# Dimensões do tabuleiro: número de cantos internos em largura e altura.
# Um tabuleiro com 9x7 quadrados tem 8x6 cantos internos.
CHECKERBOARD = (8, 6) 

# Critérios de término para o refinamento dos cantos.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- 2. Preparação dos Pontos de Objeto e Pontos de Imagem ---

objpoints = [] # Pontos 3D no espaço do mundo real
imgpoints = [] # Pontos 2D no plano da imagem

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# --- 3. Processamento das Imagens de Calibração ---

images = glob.glob('images_webcam_chess/*.png') # Altere o caminho se necessário

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow('Imagem com Cantos Detectados', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# --- 4. Calibração da Câmera ---

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# --- 5. Apresentação dos Resultados ---

print("Calibração concluída com sucesso!")
print("\nMatriz da Câmera (K):")
print(mtx)
print("\nCoeficientes de Distorção (dist):")
print(dist)
print("\nVetores de Rotação (rvecs - um por imagem):")
print(f"Total de {len(rvecs)} vetores de rotação.")
print("\nVetores de Translação (tvecs - um por imagem):")
print(f"Total de {len(tvecs)} vetores de translação.")

# --- 6. Cálculo do Erro de Reprojeção ---

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"\nErro total médio de reprojeção: {mean_error / len(objpoints)}")

# --- 7. Salvando os parâmetros de calibração em um arquivo ---

with open("calibracao_camera.pkl", "wb") as f:
    pickle.dump((mtx, dist, rvecs, tvecs), f)

print("\nParâmetros de calibração salvos em 'calibracao_camera.pkl'")

