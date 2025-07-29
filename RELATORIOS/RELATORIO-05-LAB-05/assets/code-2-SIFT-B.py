import cv2 as cv
import numpy as np
import subprocess

MIN_MATCH_COUNT = 30

# Carregar imagem de referência
img_ref_gray = cv.imread('item.png', cv.IMREAD_GRAYSCALE)
if img_ref_gray is None:
    raise Exception("Imagem de referência 'item.png' não encontrada.")
img_ref_color = cv.cvtColor(img_ref_gray, cv.COLOR_GRAY2BGR)

# Inicializar SIFT e extrair keypoints da imagem de referência
sift = cv.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(img_ref_gray, None)

# Inicializar FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Inicializar webcams
cap_left = cv.VideoCapture(0)
cap_right = cv.VideoCapture(2)

if not cap_left.isOpened() or not cap_right.isOpened():
    raise Exception("Não foi possível acessar uma ou ambas as webcams.")

# Redimensionar imagem de referência para combinar com o tamanho das câmeras
img_ref_color = cv.resize(img_ref_color, (400, 480))

# Pegar resolução das câmeras
frame_width = 640
frame_height = 480

# Tamanho final da imagem combinada
output_width = 400 + frame_width * 2
output_height = frame_height

# Inicializar gravação
fourcc = cv.VideoWriter_fourcc(*'XVID')
fps = 12.0
out = cv.VideoWriter('video-sift.avi', fourcc, fps, (output_width, output_height))

print("Gravando automaticamente... Pressione 'q' para sair.")

while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        print("Erro ao capturar vídeo.")
        break

    frame_left = cv.resize(frame_left, (frame_width, frame_height))
    frame_right = cv.resize(frame_right, (frame_width, frame_height))

    gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

    # Detectar keypoints nas câmeras
    kp_left, des_left = sift.detectAndCompute(gray_left, None)
    kp_right, des_right = sift.detectAndCompute(gray_right, None)

    if des_left is None or des_right is None:
        continue

    # Função para encontrar boas correspondências com a imagem de referência
    def get_good_matches(des_cam, kp_cam):
        matches = flann.knnMatch(des_ref, des_cam, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        matched_kp_indices = [m.trainIdx for m in good]
        matched_kps = [kp_cam[i] for i in matched_kp_indices]
        matched_desc = des_cam[matched_kp_indices] if len(matched_kp_indices) > 0 else None
        return matched_kps, matched_desc

    good_kp_left, good_des_left = get_good_matches(des_left, kp_left)
    good_kp_right, good_des_right = get_good_matches(des_right, kp_right)

    if good_des_left is None or good_des_right is None:
        continue

    matches = flann.knnMatch(good_des_left, good_des_right, k=2)
    final_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Desenhar correspondências entre câmera esquerda e direita
    img_matches = cv.drawMatches(
        frame_left, good_kp_left,
        frame_right, good_kp_right,
        final_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=2
    )

    # Combinar imagem de referência + correspondência
    combined = cv.hconcat([img_ref_color, img_matches])

    # Mostrar imagem
    cv.imshow("Referencia e Cameras", combined)

    # Gravar imagem combinada
    out.write(combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Encerrar recursos
cap_left.release()
cap_right.release()
out.release()
cv.destroyAllWindows()

# Caminho do arquivo de entrada e saída
input_file = 'video-sift.avi'
output_file = 'video-sift.mp4'

# Comando FFmpeg
command = [
    'ffmpeg',
    '-i', input_file,
    '-vcodec', 'libx264',
    '-crf', '18',             
    '-preset', 'medium',   
    output_file
]

# Executar comando
try:
    subprocess.run(command, check=True)
    print("Conversão concluída com sucesso.")
except subprocess.CalledProcessError as e:
    print(f"Erro durante a conversão: {e}")

