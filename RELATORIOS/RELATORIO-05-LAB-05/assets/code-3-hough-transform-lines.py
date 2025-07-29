import cv2
import numpy as np

# Leitura da imagem
img = cv2.imread('pt-3.0-lanes.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('pt-3.0-lanes2.jpg', cv2.IMREAD_COLOR)
img3 = cv2.imread('pt-3.0-lanes3.png', cv2.IMREAD_COLOR)

# Conversão para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Detectar bordas com Canny
edges = cv2.Canny(gray, 50, 200)
edges2 = cv2.Canny(gray2, 50, 200)
edges3 = cv2.Canny(gray3, 50, 200)

# Parâmetros da HoughLinesP
rho = 1                # Resolução em pixels
theta = np.pi / 180    # Resolução angular em radianos
threshold = 100        # Número mínimo de interseções
min_line_length = 10   # Comprimento mínimo da linha
max_line_gap = 250     # Distância máxima entre segmentos conectáveis

# Detectar linhas com HoughLinesP
lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
lines2 = cv2.HoughLinesP(edges2, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
lines3 = cv2.HoughLinesP(edges3, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

# Desenhar as linhas detectadas
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

if lines2 is not None:
    for line in lines2:
        x1, y1, x2, y2 = line[0]
        cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)

if lines3 is not None:
    for line in lines3:
        x1, y1, x2, y2 = line[0]
        cv2.line(img3, (x1, y1), (x2, y2), (255, 0, 0), 2)


# Exibir o resultado
cv2.imshow("Linhas detectadas imagem 1", img)
cv2.imshow("Linhas detectadas imagem 2", img2)
cv2.imshow("Linhas detectadas imagem 3", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

