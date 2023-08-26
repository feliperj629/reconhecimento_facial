import cv2
import mediapipe as mp

def reconhecimento_rosto():
    # Inicia a captura de vídeo da câmera
    cap = cv2.VideoCapture(1)

    # Inicializa o módulo de detecção de rosto do Mediapipe
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    while cap.isOpened():
        # Captura o próximo frame da câmera
        ret, frame = cap.read()
        if not ret:
            continue

        # Processa o frame com o Mediapipe em formato RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # Desenha retângulos em volta dos rostos detectados
        if results.detections:
            for detection in results.detections:
                caixa_delimitadora = detection.location_data.relative_bounding_box
                altura, largura, _ = frame.shape
                x, y, w, h = int(caixa_delimitadora.xmin * largura), int(caixa_delimitadora.ymin * altura), int(caixa_delimitadora.width * largura), int(caixa_delimitadora.height * altura)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Exibe o frame com os retângulos desenhados
        cv2.imshow('Reconhecimento de Rosto', frame)

        # Encerra o programa se a tecla 'q' for pressionada
        if cv2.waitKey(1) == 27:  # 27 é o código da tecla "Esc"
            break

    # Libera os recursos e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()

# Chama a função de reconhecimento de rosto
reconhecimento_rosto()
