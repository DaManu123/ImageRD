# ImageRD — Contexto Completo del Proyecto

## Descripción General

**ImageRD** es un sistema de reconocimiento óptico de caracteres (OCR) profesional
desarrollado en Python. Extrae texto de imágenes (capturas de pantalla, fotos de
documentos, escaneos) y lo exporta en múltiples formatos preservando la estructura
visual original del documento.

El sistema está especialmente optimizado para leer **capturas de exámenes tipo test**
con radio buttons (○), checkboxes (☐/☑) y opciones de respuesta, típicos de
plataformas educativas web.

---

## Arquitectura del Sistema

Pipeline lineal de 4 etapas orquestado por CLI/GUI:

```
main.py / gui.py
    │
    ▼
image_processing.py      ← Preprocesamiento de la imagen
    │
    ▼
ocr_engine.py            ← Extracción OCR con Tesseract (multi-paso)
    │
    ▼
layout_reconstructor.py  ← Reconstrucción de la estructura visual
    │
    ▼
exporters.py             ← Exportación a TXT / JSON / DOCX
```

**Módulos transversales:**
- `utils.py` — Logging, validación, detección de Tesseract, rutas
- `tesseract_manager.py` — Descarga y gestión de Tesseract bundled

### Flujo de Datos

```
str (ruta imagen)
    → np.ndarray (variantes preprocesadas)
        → OCRResult (dataclass con bloques/líneas/palabras)
            → ReconstructedDocument (texto con layout preservado)
                → Path (archivos exportados en resultados/)
```

---

## Archivos del Proyecto

| Archivo | Líneas | Responsabilidad |
|---|---|---|
| `main.py` | 384 | CLI (argparse), pipeline `run_pipeline()`, manejo de excepciones con códigos de salida |
| `gui.py` | 944 | Interfaz gráfica con CustomTkinter, vista previa de imagen, área de texto, estadísticas, barra de progreso |
| `image_processing.py` | 586 | Preprocesamiento de imágenes con OpenCV/Pillow: upscaling, CLAHE, sharpening, deskew, variantes de examen |
| `ocr_engine.py` | 1332 | Motor OCR multi-paso con ThreadPoolExecutor, post-procesamiento regex, detección de radio buttons |
| `layout_reconstructor.py` | 337 | Detección de columnas por clustering de centro-X, orden de lectura, indentación píxel→caracteres |
| `exporters.py` | 406 | Patrón dispatcher `export_all()` → TXT/JSON/DOCX. JSON con `ensure_ascii=False` y claves en español |
| `utils.py` | 256 | Logger singleton, validación de archivos, `find_tesseract()` multiplataforma, `generate_output_path()` |
| `tesseract_manager.py` | 1145 | Descarga Tesseract desde conda-forge (Linux/macOS) o UB-Mannheim (Windows), gestión de idiomas |
| **Total** | **5390** | |

---

## Tecnologías y Dependencias

### Lenguaje y Versión
- **Python 3.10+** (desarrollado/probado en Python 3.14.3)
- Sistema operativo principal: **Linux** (CachyOS / Arch-based)

### Dependencias (requirements.txt)
```
pytesseract>=0.3.10      # Wrapper Python para Tesseract OCR
opencv-python>=4.8.0     # Procesamiento de imágenes (cv2)
Pillow>=10.0.0           # Manipulación de imágenes (fallback y formatos)
numpy>=1.24.0            # Operaciones matriciales para imágenes
python-docx>=0.8.11      # Generación de documentos Word (.docx)
customtkinter>=5.2.0     # Interfaz gráfica moderna (basada en tkinter)
```

### Tesseract OCR
- **Versión:** 5.2.0 (bundled desde conda-forge)
- **Motor:** LSTM (OEM 3) — redes neuronales, no el motor legacy
- **Ubicación:** `vendor/tesseract/` (descargado automáticamente)
- **Idiomas incluidos:** `spa` (español), `eng` (inglés), `osd` (detección de orientación)
- **Configuración forzada:** `--dpi 300` en toda llamada a Tesseract

---

## Módulos en Detalle

### 1. main.py — Punto de Entrada CLI

Proporciona la interfaz de línea de comandos con argparse.

**Argumentos principales:**
| Argumento | Default | Descripción |
|---|---|---|
| `--image, -i` | *(requerido)* | Ruta a la imagen de entrada |
| `--output, -o` | `txt` | Formato: txt, json, docx, all |
| `--lang, -l` | `spa` | Idioma(s) OCR (ej: `spa+eng`) |
| `--psm` | `3` | Modo de segmentación de página |
| `--min-confidence` | `5.0` | Confianza mínima (0-100) |
| `--single-pass` | `false` | Desactiva multi-paso (más rápido) |
| `--workers, -w` | `0` (auto) | Hilos para procesamiento paralelo |
| `--no-preprocess` | `false` | Salta el preprocesamiento |
| `--verbose, -v` | `false` | Modo debug |
| `--setup-tesseract` | — | Descarga Tesseract al proyecto |
| `--add-lang` | — | Descarga idioma adicional |

**Función principal:** `run_pipeline()` ejecuta los 4 pasos secuencialmente:
1. Validación de imagen → 2. OCR multi-paso → 3. Reconstrucción → 4. Exportación

**Salida en consola:** formato decorado con Unicode (`═`, `─`, `▸`, `✓`, `✗`, `│`)
para una presentación profesional en terminal.

---

### 2. gui.py — Interfaz Gráfica

Interfaz visual moderna construida con CustomTkinter.

**Características:**
- Panel lateral con todas las opciones del CLI
- Vista previa de la imagen seleccionada
- Área de texto con scroll para el resultado OCR
- Botón de copiar texto rápido al portapapeles
- Estadísticas en tiempo real (palabras, confianza, bloques)
- Barra de progreso durante el procesamiento
- Modo oscuro/claro (toggle)
- Detección automática de Tesseract al iniciar (ofrece descarga si falta)

**Ejecución:** `python gui.py`

---

### 3. image_processing.py — Preprocesamiento de Imágenes

Prepara las imágenes para obtener máxima precisión con Tesseract LSTM.

**Principio fundamental:** El motor LSTM (OEM 3) de Tesseract funciona **mejor
con escala de grises de alta resolución** que con imágenes binarizadas. La
binarización destruye información que la red neuronal necesita.

**Pipeline de preprocesamiento:**
1. **Carga inteligente** — `smart_load()`: OpenCV con fallback a Pillow
2. **Escala de grises** — `convert_to_grayscale()`
3. **Upscaling agresivo** — `_smart_upscale()`: factor 3x para imágenes <8MP
   *(principal factor de precisión)*
4. **Corrección de inclinación** — `_deskew()`: rotación por ángulo de sesgo
5. **Mejora de contraste** — `_clahe()`: CLAHE con parámetros suaves
6. **Afinado de bordes** — `_sharpen_gentle()` / `_sharpen_strong()`
7. **Padding blanco** — `add_border_padding()`: margen de 80px

**Detección automática screenshot vs foto:**
- `_detect_if_screenshot()`: usa varianza Laplaciana (>300) + entropía Shannon (<6.8)
- Screenshots → procesamiento suave (5 variantes base + 3 de examen)
- Fotos → procesamiento fuerte (7 variantes con binarización)

**Variantes de preprocesamiento** (`generate_preprocessing_variants()`):

| Variante | Descripción | Tipo |
|---|---|---|
| V0 | Pipeline completo (todos los pasos) | Universal |
| V1 | Solo sharpen | Universal |
| V2 | Solo CLAHE | Universal |
| V3 | Upscale puro (cero procesamiento) | Universal |
| V4 | CLAHE + sharpen | Universal |
| V5 | GaussianBlur + binarización Otsu | Examen/Screenshot |
| V6 | GaussianBlur + threshold adaptativo | Examen/Screenshot |
| V7 | Limpieza morfológica CLOSE + OPEN | Examen/Screenshot |
| V5-V6 (fotos) | Otsu y adaptativo para fotos | Solo fotos |

---

### 4. ocr_engine.py — Motor OCR Multi-Paso

Núcleo del sistema. Ejecuta múltiples variantes en paralelo y selecciona
el mejor resultado por scoring.

**Clase principal: `OCREngine`**

**Estrategia multi-paso:**
1. Genera variantes de preprocesamiento (5-8 para screenshots, 7 para fotos)
2. Prueba cada variante con PSMs candidatos [3, 6, 4]
3. Ejecuta combinaciones en paralelo con `ThreadPoolExecutor`
4. Califica cada resultado con scoring: `palabras_reales × confianza_mediana`
5. Selecciona la combinación con mejor score

**Página Segmentation Modes (PSM) probados:**
- PSM 3: Segmentación automática completa
- PSM 6: Bloque uniforme de texto
- PSM 4: Columna de texto de tamaño variable

**Data classes:**
```python
@dataclass WordData     # Una palabra: texto, confianza, bounding box
@dataclass LineData     # Una línea: lista de WordData
@dataclass BlockData    # Un bloque: lista de LineData + bounding box
@dataclass OCRResult    # Resultado completo: bloques, estadísticas, metadatos
```

**Post-procesamiento (funciones clave):**

| Función | Qué hace |
|---|---|
| `_parse_words()` | Parsea datos de pytesseract, detecta radio buttons por geometría (bounding box cuadrado, ancho <60px), filtra artefactos ©®™¢ |
| `_split_radio_prefix()` | Separa "O"/"o"/"0" concatenados al texto de opciones, usa whitelist de bigramas y palabras españolas para evitar falsos positivos |
| `_fix_ocr_text()` | Normaliza comillas tipográficas → rectas, radio buttons (O/o/0→○) vía `_split_radio_prefix()`, checkboxes ([x]→☑, []→☐) |
| `_post_process_document()` | Detecta formato examen (Pregunta N, 1., Q1:), agrega separadores ─── entre preguntas, coloca ○ en opciones |
| `_apply_global_fixes()` | Corrige Ist→lst, esun→es un, acentos falsos, confusiones número/letra, limpia ruido |
| `_clean_option_text()` | Elimina prefijos residuales de opciones (a), b), O, ○ duplicado) |
| `_is_code_line()` | Detecta líneas de código Python/Java/C++/C# por patrones regex |
| `_parse_exam_questions()` | Parser de 3 fases: enunciado → código → opciones |
| `_normalize_option_lines()` | Normaliza opciones cuando no se detectan preguntas explícitas |

**Scoring:**
```
score = real_words × median_confidence
```
- `real_words`: palabras con ≥2 caracteres o en `VALID_SINGLE` (a, e, o, y, ó, í, é)
- Favorece **recall** (cantidad de texto) sobre precisión pura

---

### 5. layout_reconstructor.py — Reconstrucción de Layout

Analiza la distribución espacial de los bloques de texto.

**Clase principal: `LayoutReconstructor`**

**Algoritmo:**
1. **Detección de columnas** — Agrupa bloques por posición horizontal (centro-X)
   usando clustering con umbral configurable
2. **Orden de lectura** — Ordena bloques: columna izquierda → derecha,
   dentro de cada columna: arriba → abajo
3. **Formato de texto** — Preserva indentación (píxeles → caracteres),
   detecta saltos de párrafo por gap vertical
4. **Post-procesamiento** — Llama a `_post_process_document()` para
   formatear exámenes y aplicar correcciones globales

**Data classes:**
- `Column`: columna visual con ID, posición X y lista de bloques
- `ReconstructedDocument`: texto formateado + metadatos de estructura

---

### 6. exporters.py — Exportadores de Resultados

Genera archivos de salida en tres formatos.

**Patrón dispatcher:** `export_all()` recibe el formato ("txt"/"json"/"docx"/"all")
y delega a los exportadores específicos.

| Formato | Función | Detalles |
|---|---|---|
| `.txt` | `export_to_txt()` | Texto plano con estructura preservada |
| `.json` | `export_to_json()` | JSON con `ensure_ascii=False`, claves en español (`"confianza_promedio"`, `"total_bloques"`, etc.) |
| `.docx` | `export_to_docx()` | Documento Word formateado con fuentes, márgenes y estilos profesionales |

**Ruta de salida:** `resultados/{nombre_imagen}_{YYYY-MM-DD_HH-MM-SS}.{ext}`

---

### 7. utils.py — Utilidades Generales

Módulo transversal con funciones de soporte.

**Funciones principales:**
- `setup_logger()` — Logger singleton con formato `HH:MM:SS │ LEVEL │ mensaje`
- `validate_image_path()` — Valida existencia y formato de imagen
- `validate_output_format()` — Valida formato de salida (txt/json/docx/all)
- `find_tesseract()` — Busca Tesseract: 1) vendor/tesseract/ → 2) PATH → 3) rutas comunes
- `generate_output_path()` — Genera ruta en `resultados/` con timestamp
- `parse_languages()` — Normaliza cadena de idiomas (`"spa+eng"`)

**Formatos de imagen soportados:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.tiff`, `.tif`, `.bmp`

---

### 8. tesseract_manager.py — Gestor de Tesseract Integrado

Permite que el proyecto funcione sin instalar Tesseract como paquete del sistema.

**Funcionalidad:**
- Descarga binarios de Tesseract desde **conda-forge** (Linux/macOS) o **UB-Mannheim** (Windows)
- Extrae paquetes `.tar.bz2` (conda) o `.zip` (Windows) en `vendor/tesseract/`
- Resuelve automáticamente dependencias de librerías compartidas (libjpeg, libtiff, libxml2, lerc, icu, libiconv, openssl)
- Descarga datos de idioma (tessdata) desde GitHub `tesseract-ocr/tessdata_fast`
- Configura `LD_LIBRARY_PATH` y `TESSDATA_PREFIX` en tiempo de ejecución

**Funciones públicas:**
- `setup(languages, force)` — Descarga completa de Tesseract + idiomas
- `ensure_tesseract()` — Verifica que esté listo, descarga si falta
- `get_tesseract_path()` — Retorna Path al binario
- `configure_environment()` — Configura variables de entorno
- `add_language(lang)` — Descarga un idioma adicional

**Estructura de vendor/:**
```
vendor/tesseract/
    ├── bin/tesseract          # Binario ejecutable
    ├── lib/                   # Librerías compartidas (.so)
    ├── share/tessdata/        # Datos de idioma (.traineddata)
    └── .cache/                # Caché de descargas
```

---

## Convenciones del Código

### Bilingüe
- **Identificadores** en inglés: `image_path`, `word_count`, `min_confidence`
- **Strings de usuario, docstrings, comentarios, claves JSON y CLI** en español:
  `"confianza_promedio"`, `"total_bloques"`, `"Procesando imagen..."`

### Nomenclatura
- Constantes: `UPPER_SNAKE_CASE` (`BORDER_PADDING`, `CANDIDATE_PSMS`)
- Data classes: `PascalCase` (`WordData`, `LineData`, `BlockData`, `OCRResult`)
- Funciones públicas: `snake_case` (`preprocess_image`, `extract`)
- Funciones privadas: prefijo `_` (`_smart_upscale`, `_fix_ocr_text`)
- Separadores de sección: banners `# ─────────────────`

### Docstrings
Estilo Google con descripciones en español. Toda función pública lleva
`Args:`, `Returns:`, `Raises:`.

### UI en Terminal
Caracteres Unicode decorativos para presentación profesional:
`═`, `─`, `▸`, `✓`, `✗`, `│`, `○`, `☐`, `☑`

---

## Decisiones Técnicas Importantes

### 1. NO binarizar por defecto
Tesseract LSTM (OEM 3) rinde mejor con escala de grises que con imágenes
binarizadas. El parámetro `binarize` en `preprocess_image()` es intencionalmente
un no-op. Excepción: variantes V5-V7 para exámenes aplican binarización selectiva.

### 2. Upscaling agresivo (3x)
Es el **factor más importante** para precisión OCR. Una imagen de 800×600
se escala a 2400×1800 antes de procesarla. Ver `_smart_upscale()`.

### 3. DPI 300 siempre forzado
Cada llamada a Tesseract usa `--dpi 300` independientemente del DPI real
de la imagen. Esto es intencional y mejora la consistencia.

### 4. Scoring por real_words × median_confidence
Favorece **cantidad de texto extraído** (recall) sobre confianza pura.
Filtra artefactos de un solo carácter que no estén en `VALID_SINGLE`.

### 5. Detección screenshot vs foto
Usa varianza Laplaciana (>300) + entropía Shannon (<6.8) para diferenciar.
Screenshots reciben procesamiento suave; fotos reciben procesamiento fuerte.

### 6. Tesseract bundled
El binario de Tesseract se incluye dentro del proyecto (`vendor/tesseract/`)
para eliminar dependencias externas. Se descarga automáticamente desde
conda-forge al primer uso.

### 7. Parámetros ghost
- `binarize` en `preprocess_image()`: existe por compatibilidad, no hace nada
- `suffix` en `generate_output_path()`: existe por compatibilidad, no se usa

---

## Cómo Ejecutar

### Requisitos Previos
```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Instalar dependencias
pip install -r requirements.txt

# Tesseract se descarga automáticamente al primer uso, o manualmente:
python main.py --setup-tesseract
```

### Uso por Terminal (CLI)
```bash
# Uso básico
python main.py --image documento.jpg --output txt --lang spa

# Todos los formatos
python main.py --image foto.png --output all --lang spa+eng

# Modo rápido (sin multi-paso)
python main.py --image captura.png --output txt --single-pass

# Con debug
python main.py --image scan.tiff --output json --verbose

# Agregar idioma
python main.py --add-lang fra
```

### Uso con Interfaz Gráfica
```bash
python gui.py
```

### Resultados
Los archivos generados se guardan en:
```
resultados/{nombre_imagen}_{YYYY-MM-DD_HH-MM-SS}.{ext}
```

---

## Historial de Desarrollo

| Commit | Descripción |
|---|---|
| `afa7858` | feat: Sistema OCR completo ImageRD v1.0.0 |
| `e7bfdbb` | refactor: corrección de errores IDE y mejoras de proyecto |
| `8d7e046` | feat: GUI CustomTkinter + post-procesamiento inteligente para exámenes |
| `b8dc56c` | perf: procesamiento OCR paralelo con ThreadPoolExecutor |
| `ecf293c` | feat: auto-PSM, GUI fix guardado resultados/, max extracción texto |
| `cbc10c8` | feat: Tesseract OCR integrado dentro del proyecto (sin instalación externa) |
| `88f7dbf` | feat: mejoras OCR para lectura de exámenes con radio buttons |
| `db97ed9` | fix: eliminar falsos positivos en detección de radio buttons OCR |

---

## Al Modificar Código

- **Nuevas variantes de preprocesamiento:** agregarlas en `generate_preprocessing_variants()` de `image_processing.py`
- **Nuevos formatos de exportación:** seguir el patrón de `export_to_txt/json/docx` y registrar en `export_all()`
- **Post-procesamiento OCR:** agregar patrones en `_fix_ocr_text()` de `ocr_engine.py`
- **Correcciones globales:** agregar en `_apply_global_fixes()` de `ocr_engine.py`
- **Nuevos patrones de código:** agregar en `_is_code_line()` de `ocr_engine.py`
- **Parámetros ghost:** `binarize` en `preprocess_image()` y `suffix` en `generate_output_path()` existen por compatibilidad — **no eliminar**
- **No hay test suite ni CI.** Validar cambios ejecutando el pipeline contra una imagen de prueba
