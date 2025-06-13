# TecladoVirtual
ejemplo: de un teclado virtual manipulado con los ojos en python
Tabla de Contenidos
Descripción del Proyecto

Requisitos del Sistema

Instalación

Configuración Inicial

Ejecución del Programa

Calibración

Uso del Sistema

Personalización

Solución de Problemas

Contribución

Licencia

Descripción del Proyecto
Sistema de escritura asistida que permite a usuarios con movilidad reducida comunicarse mediante el seguimiento ocular. Utiliza una cámara web convencional para detectar la mirada y seleccionar caracteres en un teclado virtual, con funcionalidades de autocompletado predictivo.

Características principales:

🎯 Seguimiento ocular preciso con calibración personalizada

⌨️ Teclado virtual con distribución QWERTY optimizada

💡 Autocompletado inteligente basado en diccionario

🔊 Retroalimentación visual y auditiva

💾 Guardado automático del texto generado

Requisitos del Sistema
Hardware
Cámara web (integrada o externa)

Procesador: Intel i3 o equivalente (mínimo)

Memoria RAM: 4GB (recomendado 8GB)

Espacio en disco: 100MB

Software
Sistema Operativo: Windows 10/11 (compatible con Linux/macOS con ajustes)

Python: Versión 3.8 o superior

Librerías: OpenCV, MediaPipe, NumPy

Instalación
1. Clonar el repositorio
bash
git clone https://github.com/tu-usuario/sistema-escritura-ocular.git
cd sistema-escritura-ocular
2. Crear entorno virtual (Recomendado)
bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
3. Instalar dependencias
bash
pip install -r requirements.txt
4. Verificar instalación
bash
python -c "import cv2; print(cv2.__version__)"
Configuración Inicial
Archivo de configuración (config.py)
python
# Tiempo de permanencia para selección (en segundos)
DWELL_TIME = 1.5

# Umbral de precisión ocular (ajustar según necesidad)
EAR_THRESHOLD = 0.18

# Resolución de cámara recomendada
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
Diccionario de palabras (palabras.txt)
text
HOLA
MUNDO
PYTHON
PROGRAMA
TECLADO
... # Añadir palabras personalizadas
Ejecución del Programa
En Windows
bash
python main.py
En Linux/macOS
bash
python3 main.py
Calibración
Posición inicial:

Siéntese a 50-70 cm de la cámara

Asegure buena iluminación frontal

Proceso de calibración:

Siga las instrucciones en pantalla

Mire fijamente los puntos rojos que aparecerán

Cada punto requiere 2 segundos de mirada

El sistema emitirá sonidos de confirmación

Re-calibración:

Presione 'R' durante la ejecución para reiniciar calibración

Uso del Sistema
Interfaz principal:
text
+---------------------------------+
| Texto actual: [Área de edición] |
| Texto enviado: [Texto confirmado] |
+---------------------------------+
| [Teclado QWERTY con 4 filas]    |
+---------------------------------+
| [Sugerencias predictivas]       |
+---------------------------------+
Funciones clave:
Selección de caracteres:

Mire una tecla durante 1.5 segundos para seleccionarla

Barra de progreso verde indica tiempo de selección

Teclas especiales:

<: Borrar último carácter

ESPACIO: Insertar espacio

ENTER: Confirmar texto

Autocompletado:

Las sugerencias aparecen automáticamente

Mire una sugerencia para seleccionarla

Tiempo de selección igual que para teclas

Guardado automático:

Al cerrar el programa, el texto se guarda en:

texto_escritura.txt

Atajos de teclado:
ESC: Salir del programa

R: Reiniciar calibración

C: Limpiar texto actual

Personalización
Añadir palabras al diccionario
Abra el archivo palabras.txt

Añada nuevas palabras en mayúsculas, una por línea

Guarde el archivo

Reinicie la aplicación para cargar cambios

Modificar diseño del teclado
Edite la constante KEYBOARD en el código:

python
KEYBOARD = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', '<'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '_'],
    ['SPACE', 'ENTER']
]
Ajustar parámetros de rendimiento
Modifique en config.py:

python
# Reducir para mayor velocidad, aumentar para precisión
SMOOTHING_WINDOW = 150

# Umbral de movimiento mínimo (píxeles)
MOVEMENT_THRESHOLD = 25
Solución de Problemas
Problema: Cámara no detectada
Solución:

Verifique conexión física

Asegúrese que no hay otras aplicaciones usando la cámara

Especifique índice de cámara manualmente (línea 420 en main.py):

python
cap = cv2.VideoCapture(0)  # Cambiar 0 por 1, 2, etc.
Problema: Seguimiento impreciso
Solución:

Mejore la iluminación ambiental

Elimine reflejos en gafas

Realice nueva calibración (presione 'R')

Ajuste EAR_THRESHOLD en config.py

Problema: Rendimiento lento
Solución:

Cierre otras aplicaciones

Reduzca resolución de cámara (línea 422 en main.py):

python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
Disminuya SMOOTHING_WINDOW en config.py

Contribución
¡Bienvenidas contribuciones! Pasos para contribuir:

Realice un fork del repositorio

Cree una nueva rama (git checkout -b feature/nueva-funcion)

Realice sus cambios y commit (git commit -m 'Añade nueva función')

Push a la rama (git push origin feature/nueva-funcion)

Abra un Pull Request

Áreas prioritarias para desarrollo:

Soporte multi-idioma

Integración con sintetizadores de voz

Modo de navegación por pestañas

