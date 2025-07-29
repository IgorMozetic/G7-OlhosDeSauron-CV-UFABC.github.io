import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

# Carregar as imagens em escala de cinza
img1 = cv.imread('pt-2.0-item.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('pt-2.1-item-na-imagem.png', cv.IMREAD_GRAYSCALE)  # trainImage

# Iniciar o detector SIFT
sift = cv.SIFT_create()

# Encontrar os keypoints e descritores com o SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Parâmetros para o FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Inicializar o FLANN-based Matcher
flann = cv.FlannBasedMatcher(index_params, search_params)

# Realizar a correspondência dos descritores
matches = flann.knnMatch(des1, des2, k=2)

# Armazenar todas as boas correspondências conforme o teste de razão de Lowe
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# Verificar se as correspondências são suficientes para calcular a homografia
if len(good) > MIN_MATCH_COUNT:
    # Obter os pontos correspondentes nas duas imagens
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Calcular a homografia usando RANSAC
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Transformar a perspectiva
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    # Desenhar o polígono da transformação de perspectiva
    img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    # Desenhar as boas correspondências
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **{
        'matchColor': (0, 255, 0),  # Cor das correspondências
        'singlePointColor': (255, 0, 0),  # Cor dos pontos-chave
        'flags': 2
    })

    # Exibir a imagem com as correspondências e a transformação
    plt.imshow(img3)
    plt.show()

else:
    print("Não há correspondências suficientes - {}/{}".format(len(good), MIN_MATCH_COUNT))

