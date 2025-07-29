import cv2
import numpy as np
import subprocess

# Inicializar as duas webcams (substitua os índices se necessário)
cap_left = cv2.VideoCapture(2)
cap_right = cv2.VideoCapture(0)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Erro: não foi possível abrir as câmeras.")
    exit()

# Parâmetros da HoughLinesP
rho = 1
theta = np.pi / 180
threshold = 100
min_line_length = 10
max_line_gap = 250

# Definir o codec e criar o objeto VideoWriter para salvar a gravação
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  # ou use cap_left.get(cv2.CAP_PROP_FPS) para pegar o FPS real da câmera
frame_width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter para salvar a gravação
out = cv2.VideoWriter('video-hough-lines.avi', fourcc, fps, (frame_width * 2, frame_height))

print("Pressione 'q' para sair.")

while True:
    # Capturar frames das duas câmeras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Erro ao capturar imagens das câmeras.")
        break

    def process_frame(frame):
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detectar bordas
        edges = cv2.Canny(gray, 50, 200)
        # Detectar linhas
        lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        # Desenhar linhas
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame

    # Processar os dois frames
    processed_left = process_frame(frame_left.copy())
    processed_right = process_frame(frame_right.copy())

    # Redimensionar para exibição lado a lado (opcional)
    processed_left = cv2.resize(processed_left, (640, 480))
    processed_right = cv2.resize(processed_right, (640, 480))

    # Combinar os dois vídeos lado a lado
    combined = np.hstack((processed_left, processed_right))

    # Gravar o frame combinado
    out.write(combined)

    # Exibir
    cv2.imshow('Linhas Detectadas - Direita | Esquerda', combined)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap_left.release()
cap_right.release()
out.release()
cv2.destroyAllWindows()


# Caminho do arquivo de entrada e saída
input_file = 'video-hough-lines.avi'
output_file = 'video-hough-lines.mp4'

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

