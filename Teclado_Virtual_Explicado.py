import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import os
import statistics

# Inicialización de módulos de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constantes de configuración
EAR_THRESHOLD = 0.18         # Umbral para detección de parpadeo
CONSECUTIVE_FRAMES = 2       # Frames consecutivos para confirmar acción
KEY_SIZE = 60                # Tamaño base de las teclas
KEY_MARGIN = 5               # Espacio entre teclas
SCREEN_WIDTH = 1280          # Ancho de la ventana
SCREEN_HEIGHT = 720          # Alto de la ventana
TEXT_AREA_HEIGHT = 100       # Altura del área de texto
KEYBOARD_Y = TEXT_AREA_HEIGHT + 50  # Posición Y del teclado
DWELL_TIME = 1.5             # Tiempo de permanencia para selección

# Distribución del teclado virtual
KEYBOARD = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', '<'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '_'],
    ['SPACE', 'ENTER']  # Teclas especiales
]

# Índices de landmarks oculares (MediaPipe)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

class EyeTracker:
    def __init__(self):
        # Inicialización del modelo FaceMesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Historial para suavizado de movimiento
        self.cursor_pos_history = []
        self.SMOOTHING_WINDOW = 150     # Tamaño de ventana para suavizado
        self.MOVEMENT_THRESHOLD = 25    # Umbral mínimo de movimiento
        # Variables para calibración
        self.calibration_points = []
        self.calibration_complete = False
        self.calibration_stage = 0
        self.calibration_start_time = 0
        # Estado del cursor
        self.last_valid_cursor = (SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
        self.stability_counter = 0
        self.last_update_time = time.time()
        self.cursor_stable_pos = self.last_valid_cursor

    def calculate_ear(self, landmarks, eye_indices):
        """Calcula el Eye Aspect Ratio (EAR) para detectar parpadeos"""
        # Obtener coordenadas de los puntos oculares
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        
        # Calcular distancias necesarias
        dist1 = np.linalg.norm(p2 - p6)
        dist2 = np.linalg.norm(p3 - p5)
        dist3 = np.linalg.norm(p1 - p4)
        
        # Fórmula EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        return (dist1 + dist2) / (2.0 * dist3)

    def get_iris_center(self, landmarks, iris_indices):
        """Calcula el centro del iris a partir de los landmarks"""
        points = np.array([(landmarks[i].x, landmarks[i].y) for i in iris_indices])
        return np.mean(points, axis=0)
    
    def calibrate(self, frame, gaze_vector):
        """Realiza el proceso de calibración para mapear la mirada a la pantalla"""
        current_time = time.time()
        
        # Iniciar calibración
        if self.calibration_stage == 0:
            self.calibration_stage = 1
            self.calibration_start_time = current_time
            self.calibration_points = []
            self.calibration_targets = []
        
        # Instrucción para el usuario
        cv2.putText(frame, "Mire al punto rojo", (SCREEN_WIDTH//2 - 150, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Puntos de calibración (esquinas y centro)
        calibration_targets = [
            (SCREEN_WIDTH//4, SCREEN_HEIGHT//4),
            (3*SCREEN_WIDTH//4, SCREEN_HEIGHT//4),
            (SCREEN_WIDTH//4, 3*SCREEN_HEIGHT//4),
            (3*SCREEN_WIDTH//4, 3*SCREEN_HEIGHT//4),
            (SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
        ]
        
        # Dibujar punto actual de calibración
        target_x, target_y = calibration_targets[self.calibration_stage - 1]
        cv2.circle(frame, (target_x, target_y), 15, (0, 0, 255), -1)
        
        # Recolectar datos después de 2 segundos
        if current_time - self.calibration_start_time > 2.0:
            self.calibration_points.append(gaze_vector)
            self.calibration_targets.append((target_x / SCREEN_WIDTH, target_y / SCREEN_HEIGHT))
            
            self.calibration_stage += 1
            self.calibration_start_time = current_time
            winsound.Beep(500, 100)  # Feedback auditivo
            
            # Finalizar calibración después de 5 puntos
            if self.calibration_stage > 5:
                self.calibrate_linear_model()
                self.calibration_complete = True
                winsound.Beep(1000, 500)
                return True
        
        return False
    
    def calibrate_linear_model(self):
        """Crea modelo lineal para mapeo de mirada a coordenadas de pantalla"""
        gaze_x = [g[0] for g in self.calibration_points]
        gaze_y = [g[1] for g in self.calibration_points]
        target_x = [t[0] for t in self.calibration_targets]
        target_y = [t[1] for t in self.calibration_targets]
        
        # Crear modelo de regresión lineal
        A = np.vstack([gaze_x, gaze_y, np.ones(len(gaze_x))]).T
        self.model_x = np.linalg.lstsq(A, target_x, rcond=None)[0]
        self.model_y = np.linalg.lstsq(A, target_y, rcond=None)[0]
    
    def map_gaze_to_screen(self, gaze_vector):
        """Mapea el vector de mirada a coordenadas de pantalla usando el modelo de calibración"""
        if not self.calibration_complete:
            return None
            
        x, y = gaze_vector
        # Predecir coordenadas usando el modelo lineal
        pred_x = self.model_x[0]*x + self.model_x[1]*y + self.model_x[2]
        pred_y = self.model_y[0]*x + self.model_y[1]*y + self.model_y[2]
        
        # Convertir a píxeles y asegurar que están dentro de los límites
        screen_x = int(pred_x * SCREEN_WIDTH)
        screen_y = int(pred_y * SCREEN_HEIGHT)
        screen_x = max(0, min(SCREEN_WIDTH - 1, screen_x))
        screen_y = max(0, min(SCREEN_HEIGHT - 1, screen_y))
        
        return (screen_x, screen_y)
    
    def detect_gaze(self, frame):
        """Detecta la mirada y devuelve la posición del cursor en la pantalla"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        cursor_pos = self.last_valid_cursor  # Valor por defecto

        if results.multi_face_landmarks:
            try:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Calcular centro de los iris
                left_iris = self.get_iris_center(landmarks, LEFT_IRIS_INDICES)
                right_iris = self.get_iris_center(landmarks, RIGHT_IRIS_INDICES)
                iris_center = (left_iris + right_iris) / 2.0
                gaze_vector = (iris_center[0], iris_center[1])
                
                # Realizar calibración si es necesario
                if not self.calibration_complete:
                    self.calibrate(frame, gaze_vector)
                    return cursor_pos
                
                # Mapear mirada a pantalla
                new_cursor_pos = self.map_gaze_to_screen(gaze_vector)
                
                if new_cursor_pos:
                    # Limitar frecuencia de actualización (~20 FPS)
                    current_time = time.time()
                    if current_time - self.last_update_time < 0.05:
                        return cursor_pos
                    self.last_update_time = current_time
                    
                    # Calcular distancia desde la última posición
                    if self.last_valid_cursor:
                        dx = new_cursor_pos[0] - self.last_valid_cursor[0]
                        dy = new_cursor_pos[1] - self.last_valid_cursor[1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        # Contar frames estables si el movimiento es pequeño
                        if distance < self.MOVEMENT_THRESHOLD:
                            cursor_pos = self.last_valid_cursor
                            self.stability_counter += 1
                        else:
                            cursor_pos = new_cursor_pos
                            self.stability_counter = 0
                    else:
                        cursor_pos = new_cursor_pos
                        self.stability_counter = 0
                    
                    self.last_valid_cursor = cursor_pos
                
                # Suavizado avanzado del cursor
                if cursor_pos:
                    self.cursor_pos_history.append(cursor_pos)
                    if len(self.cursor_pos_history) > self.SMOOTHING_WINDOW:
                        self.cursor_pos_history.pop(0)
                    
                    if self.cursor_pos_history:
                        # Eliminar outliers usando mediana
                        smoothed_x = statistics.median([p[0] for p in self.cursor_pos_history])
                        smoothed_y = statistics.median([p[1] for p in self.cursor_pos_history])
                        
                        # Suavizado adicional con media móvil exponencial
                        alpha = 0.8
                        self.cursor_stable_pos = (
                            int(alpha * self.cursor_stable_pos[0] + (1 - alpha) * smoothed_x),
                            int(alpha * self.cursor_stable_pos[1] + (1 - alpha) * smoothed_y)
                        )
                        
                        # Usar posición suavizada si hay estabilidad
                        if self.stability_counter > 10:
                            cursor_pos = self.cursor_stable_pos
                        else:
                            # Interpolar para movimientos rápidos
                            beta = 0.3
                            x_coord = int(self.cursor_stable_pos[0] * beta + cursor_pos[0] * (1 - beta))
                            y_coord = int(self.cursor_stable_pos[1] * beta + cursor_pos[1] * (1 - beta))
                            cursor_pos = (x_coord, y_coord)
                        
                        self.last_valid_cursor = cursor_pos
            
            except Exception as e:
                print(f"Error en detección ocular: {e}")
                cursor_pos = self.last_valid_cursor
        else:
            cursor_pos = self.last_valid_cursor

        return cursor_pos

class VirtualKeyboard:
    def __init__(self):
        self.text = ""          # Texto en edición
        self.final_text = ""    # Texto confirmado
        self.suggestions = []   # Sugerencias de palabras
        self.keyboard = KEYBOARD
        self.keys = self.create_keys()  # Crear teclas
        # Control de tiempo para selecciones
        self.last_key_press_time = 0
        self.current_key = None
        self.key_start_time = 0
        self.current_suggestion = None
        self.suggestion_start_time = 0
        self.selection_cooldown = 0.5  # Tiempo entre selecciones
        self.last_suggestion_selected_time = 0
        # Cargar diccionario para sugerencias
        self.prediction_words = self.load_word_list("palabras.txt")
    
    def load_word_list(self, filename):
        """Carga lista de palabras para sugerencias desde archivo"""
        if not os.path.exists(filename):
            return []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return [line.strip().upper() for line in f.readlines() if line.strip()]
        except:
            return []
    
    def create_keys(self):
        """Crea la estructura de teclas con sus posiciones y tamaños"""
        keys = []
        start_y = KEYBOARD_Y
        
        for row_idx, row in enumerate(self.keyboard):
            num_keys = len(row)
            key_widths = [KEY_SIZE] * num_keys
            
            # Ajustar tamaño de teclas especiales
            if row_idx == 3:
                key_widths = [KEY_SIZE * 4, KEY_SIZE * 4]
            
            total_width = sum(key_widths) + (num_keys - 1) * KEY_MARGIN
            start_x = (SCREEN_WIDTH - total_width) // 2
            
            # Crear cada tecla con sus propiedades
            for col_idx, key in enumerate(row):
                key_x = start_x + sum(key_widths[:col_idx]) + col_idx * KEY_MARGIN
                key_y = start_y + row_idx * (KEY_SIZE + KEY_MARGIN)
                width = key_widths[col_idx]
                
                keys.append({
                    'char': key,
                    'pos': (key_x, key_y),
                    'size': (width, KEY_SIZE),
                    'selected': False,
                    'dwell_progress': 0,
                    'dwell_start': 0,
                    'last_selected': 0
                })
        
        return keys
    
    def get_selected_key(self, cursor_pos):
        """Encuentra la tecla bajo el cursor"""
        if cursor_pos is None:
            return None
            
        for key in self.keys:
            x, y = key['pos']
            w, h = key['size']
            
            if (x <= cursor_pos[0] <= x + w and 
                y <= cursor_pos[1] <= y + h):
                return key
        
        return None
    
    def handle_key_press(self, key):
        """Procesa la acción al seleccionar una tecla"""
        if key is None:
            return
            
        current_time = time.time()
        # Evitar selecciones rápidas consecutivas
        if current_time - key['last_selected'] < self.selection_cooldown:
            return
            
        winsound.Beep(440, 100)  # Feedback auditivo
        
        # Acciones según tipo de tecla
        if key['char'] == '<':
            self.text = self.text[:-1]  # Borrar
        elif key['char'] == '_' or key['char'] == 'SPACE':
            self.text += ' '  # Espacio
        elif key['char'] == 'ENTER':
            self.final_text += self.text + ' '  # Confirmar texto
            self.text = ""
        else:
            self.text += key['char']  # Caracter normal
        
        # Actualizar estado
        self.update_suggestions()
        key['selected'] = True
        key['dwell_progress'] = 0
        key['last_selected'] = current_time
        self.last_key_press_time = current_time
        self.current_key = None
    
    def update_dwell_times(self, cursor_pos):
        """Actualiza temporizadores de permanencia para teclas y sugerencias"""
        current_time = time.time()
        key = self.get_selected_key(cursor_pos)
        suggestion = self.get_suggestion_at_pos(cursor_pos)
        
        # Reiniciar si no hay cursor
        if cursor_pos is None:
            for k in self.keys:
                k['dwell_progress'] = 0
            self.current_key = None
            return
        
        # Procesar tecla seleccionada
        if key:
            # Esperar cooldown
            if current_time - key['last_selected'] < self.selection_cooldown:
                key['dwell_progress'] = 0
                self.current_key = None
            else:    
                # Iniciar/continuar temporizador
                if key != self.current_key:
                    self.current_key = key
                    key['dwell_start'] = current_time
                    key['dwell_progress'] = 0
                else:
                    elapsed = current_time - key['dwell_start']
                    key['dwell_progress'] = min(elapsed / DWELL_TIME, 1.0)
                    # Seleccionar al completar tiempo
                    if elapsed >= DWELL_TIME:
                        self.handle_key_press(key)
                        key['dwell_start'] = current_time
        else:
            self.current_key = None
            # Reiniciar todas las teclas
            for k in self.keys:
                k['dwell_progress'] = 0
        
        # Procesar sugerencia seleccionada
        if suggestion:
            if current_time - self.last_suggestion_selected_time < self.selection_cooldown:
                self.current_suggestion = None
                return
                
            if suggestion != self.current_suggestion:
                self.current_suggestion = suggestion
                self.suggestion_start_time = current_time
            else:
                elapsed = current_time - self.suggestion_start_time
                # Seleccionar al completar tiempo
                if elapsed >= DWELL_TIME:
                    self.select_suggestion(suggestion)
                    self.suggestion_start_time = current_time
        else:
            self.current_suggestion = None
    
    def update_suggestions(self):
        """Actualiza sugerencias basadas en el texto actual"""
        if not self.prediction_words or not self.text:
            self.suggestions = []
            return
            
        # Extraer palabra actual
        if ' ' in self.text:
            last_space = self.text.rfind(' ')
            current_word = self.text[last_space+1:]
        else:
            current_word = self.text
        
        current_word = current_word.upper()
        
        # Filtrar palabras que comiencen con el texto actual
        self.suggestions = [
            word for word in self.prediction_words 
            if word.startswith(current_word)
        ][:3]  # Limitar a 3 sugerencias
    
    def get_suggestion_at_pos(self, cursor_pos):
        """Encuentra sugerencia bajo el cursor"""
        if not self.suggestions or cursor_pos is None:
            return None
            
        for i, suggestion in enumerate(self.suggestions):
            # Posición de las sugerencias (lado derecho)
            x = SCREEN_WIDTH - 200
            y = 150 + i * 40
            w = 180
            h = 35
            
            if (x <= cursor_pos[0] <= x + w and 
                y <= cursor_pos[1] <= y + h):
                return suggestion
        
        return None
    
    def select_suggestion(self, suggestion):
        """Selecciona una sugerencia para autocompletar"""
        if suggestion:
            winsound.Beep(880, 100)
            self.last_suggestion_selected_time = time.time()
            
            # Reemplazar palabra actual por sugerencia
            if ' ' in self.text:
                last_space = self.text.rfind(' ')
                self.text = self.text[:last_space+1] + suggestion + ' '
            else:
                self.text = suggestion + ' '
            
            self.update_suggestions()
            self.current_suggestion = None

def draw_interface(frame, keyboard, cursor_pos):
    """Dibuja toda la interfaz de usuario en el frame"""
    # Área de texto en edición
    cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH, TEXT_AREA_HEIGHT), (50, 50, 50), -1)
    cv2.putText(frame, "Texto actual: " + keyboard.text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Área de texto confirmado
    cv2.rectangle(frame, (0, TEXT_AREA_HEIGHT), (SCREEN_WIDTH, TEXT_AREA_HEIGHT + 50), 
                 (70, 70, 70), -1)
    cv2.putText(frame, "Texto enviado: " + keyboard.final_text.strip(), (20, TEXT_AREA_HEIGHT + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Sugerencias (lado derecho)
    for i, suggestion in enumerate(keyboard.suggestions):
        x = SCREEN_WIDTH - 200
        y = 150 + i * 40
        w = 180
        h = 35
        
        # Resaltar sugerencia seleccionada
        is_selected = (cursor_pos and 
                      keyboard.get_suggestion_at_pos(cursor_pos) == suggestion)
        color = (0, 200, 255) if is_selected else (0, 150, 255)
        
        # Barra de progreso para selección
        if is_selected and keyboard.current_suggestion == suggestion:
            elapsed = time.time() - keyboard.suggestion_start_time
            progress = min(elapsed / DWELL_TIME, 1.0)
            bar_width = int(w * progress)
            cv2.rectangle(frame, (x, y), (x + bar_width, y + 5), (0, 255, 0), -1)
        
        # Dibujar recuadro y texto de sugerencia
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.putText(frame, suggestion, (x + 10, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Dibujar teclado
    for key in keyboard.keys:
        progress = key['dwell_progress']
        # Color dinámico (rojo -> verde)
        if progress > 0:
            r = int(200 * (1 - progress))
            g = int(200 * progress)
            b = int(200 * (1 - progress))
            color = (b, g, r)
        else:
            color = (200, 200, 200)
        
        # Resaltar tecla bajo cursor
        if cursor_pos and keyboard.get_selected_key(cursor_pos) == key:
            color = (0, 200, 255)
        
        x, y = key['pos']
        w, h = key['size']
        
        # Dibujar tecla
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 2)
        
        # Barra de progreso de selección
        if progress > 0:
            bar_height = 5
            bar_width = int(w * progress)
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (0, 255, 0), -1)
        
        # Texto de la tecla
        char = key['char']
        if char == 'SPACE':
            char = "ESPACIO"
        elif char == 'ENTER':
            char = "ENTER"
            
        text_size = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        cv2.putText(frame, char, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Dibujar cursor
    if cursor_pos:
        # Color dinámico (rojo = inestable, verde = estable)
        stability = min(eye_tracker.stability_counter / 10, 1.0)
        cursor_color = (0, int(255 * stability), int(255 * (1 - stability)))
        
        cv2.circle(frame, cursor_pos, 10, cursor_color, -1)
        cv2.circle(frame, cursor_pos, 12, (255, 255, 255), 2)
    
    # Reiniciar estado de teclas seleccionadas
    if time.time() - keyboard.last_key_press_time > 0.5:
        for key in keyboard.keys:
            key['selected'] = False

def main():
    global eye_tracker
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    
    # Inicializar componentes
    eye_tracker = EyeTracker()
    keyboard = VirtualKeyboard()
    
    # Crear ventana
    cv2.namedWindow('Eye Typing System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Eye Typing System', SCREEN_WIDTH, SCREEN_HEIGHT)
    
    try:
        while cap.isOpened():
            # Capturar frame
            success, frame = cap.read()
            if not success:
                print("Error de cámara")
                break
            
            frame = cv2.flip(frame, 1)  # Espejo para interfaz natural
            
            # Asegurar tamaño correcto
            if frame.shape[1] != SCREEN_WIDTH or frame.shape[0] != SCREEN_HEIGHT:
                frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # Procesamiento principal
            cursor_pos = eye_tracker.detect_gaze(frame)
            keyboard.update_dwell_times(cursor_pos)
            keyboard.update_suggestions()
            draw_interface(frame, keyboard, cursor_pos)
            
            # Mostrar resultado
            cv2.imshow('Eye Typing System', frame)
            
            # Salir con ESC
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Guardar texto al salir
        with open('texto_escrito.txt', 'w', encoding='utf-8') as f:
            f.write(keyboard.final_text + keyboard.text)
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Crear archivo de palabras si no existe
    if not os.path.exists("palabras.txt"):
        with open("palabras.txt", "w", encoding="utf-8") as f:
            f.write("HOLA\nMUNDO\nPYTHON\nPROGRAMA\nTECLADO\nACCESIBILIDAD\n")
    
    main()