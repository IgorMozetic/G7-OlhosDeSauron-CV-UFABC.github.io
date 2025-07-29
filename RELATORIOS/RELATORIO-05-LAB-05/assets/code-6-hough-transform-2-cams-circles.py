import cv2
import numpy as np
import subprocess

# Inicializar as duas webcams (substitua os índices se necessário)
cap_left = cv2.VideoCapture(2)  # Webcam esquerda
cap_right = cv2.VideoCapture(0)  # Webcam direita

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Erro: não foi possível abrir as câmeras.")
    exit()

# Definir o codec e criar o objeto VideoWriter para salvar a gravação
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  # ou use cap_left.get(cv2.CAP_PROP_FPS) para pegar o FPS real da câmera
frame_width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter para salvar a gravação
out = cv2.VideoWriter('video-hough-circles.avi', fourcc, fps, (frame_width * 2, frame_height))

# Função para detectar círculos usando Hough Transform
def detect_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
    blurred = cv2.medianBlur(gray, 5)  # Reduzir ruído com filtro de mediana
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=40, minRadius=10, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Desenhar o círculo externo
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Desenhar o centro do círculo
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    return frame

print("Pressione 'q' para sair.")

while True:
    # Capturar frames das duas câmeras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Erro ao capturar imagens das câmeras.")
        break

    # Detectar círculos nos frames das webcams
    frame_left_with_circles = detect_circles(frame_left.copy())
    frame_right_with_circles = detect_circles(frame_right.copy())

    # Redimensionar os frames para exibição lado a lado
    frame_left_resized = cv2.resize(frame_left_with_circles, (640, 480))
    frame_right_resized = cv2.resize(frame_right_with_circles, (640, 480))

    # Combinar os frames lado a lado
    combined_frame = np.hstack((frame_left_resized, frame_right_resized))

    # Gravar o frame combinado
    out.write(combined_frame)

    # Exibir o resultado
    cv2.imshow('Circulos Detectados - Esquerda | Direita', combined_frame)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap_left.release()
cap_right.release()
out.release()
cv2.destroyAllWindows()



# Caminho do arquivo de entrada e saída
input_file = 'video-hough-circles.avi'
output_file = 'video-hough-circles.mp4'

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

