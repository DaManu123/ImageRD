# ImageRD — Sistema OCR Profesional

Aplicación profesional en Python para extracción de texto de imágenes,
respetando la estructura visual original del documento.

---

## Características

- **Extracción avanzada de texto** con Tesseract OCR
- **Preprocesamiento inteligente**: escala de grises, binarización adaptativa, reducción de ruido, mejora de contraste, corrección de inclinación
- **Reconstrucción de estructura**: columnas, párrafos, bloques, saltos de línea
- **Detección de texto en imágenes embebidas** (capturas, gráficos, memes)
- **Multi-idioma**: español, inglés, y cualquier combinación soportada por Tesseract
- **Múltiples formatos de salida**: `.txt`, `.docx`, `.json`
- **CLI profesional** con argumentos configurables

---

## Requisitos previos

### Python
- Python **3.10** o superior

### Tesseract OCR
Debes instalar Tesseract OCR en tu sistema:

#### Windows
1. Descarga el instalador desde: https://github.com/UB-Mannheim/tesseract/wiki
2. Ejecuta el instalador (selecciona los idiomas que necesites, al menos **Spanish** y **English**)
3. Asegúrate de que el directorio de instalación esté en el PATH o en la ruta por defecto:
   ```
   C:\Program Files\Tesseract-OCR\
   ```

#### macOS
```bash
brew install tesseract
brew install tesseract-lang   # Para idiomas adicionales
```

#### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-spa   # Español
sudo apt install tesseract-ocr-eng   # Inglés
```

---

## Instalación

```bash
# Clonar o copiar el proyecto
cd ImageRD

# Instalar dependencias de Python
pip install -r requirements.txt
```

---

## Uso

### Comando básico
```bash
python main.py --image documento.jpg --output txt --lang spa
```

### Todos los formatos de salida
```bash
python main.py --image documento.jpg --output all --lang spa
```

### Múltiples idiomas
```bash
python main.py --image documento.jpg --output docx --lang spa+eng
```

### Modo verbose (detallado)
```bash
python main.py --image documento.jpg --output json --lang spa --verbose
```

### Sin preprocesamiento
```bash
python main.py --image foto_clara.png --output txt --no-preprocess
```

### Ajustar confianza mínima
```bash
python main.py --image baja_calidad.jpg --output txt --min-confidence 20
```

---

## Argumentos CLI

| Argumento          | Alias | Tipo   | Default | Descripción                                       |
|--------------------|-------|--------|---------|---------------------------------------------------|
| `--image`          | `-i`  | str    | —       | Ruta a la imagen de entrada (obligatorio)          |
| `--output`         | `-o`  | str    | `txt`   | Formato: `txt`, `json`, `docx`, `all`              |
| `--lang`           | `-l`  | str    | `spa`   | Idioma(s) OCR (e.g., `spa`, `eng`, `spa+eng`)     |
| `--output-path`    | `-op` | str    | —       | Ruta personalizada del archivo de salida           |
| `--min-confidence` | `-mc` | float  | `30`    | Confianza mínima (0-100)                           |
| `--psm`            |       | int    | `3`     | Modo de segmentación de página de Tesseract        |
| `--no-preprocess`  |       | flag   | —       | Desactivar preprocesamiento de imagen              |
| `--verbose`        | `-v`  | flag   | —       | Modo detallado (debug)                             |
| `--version`        | `-V`  | —      | —       | Mostrar versión                                    |

---

## Estructura del proyecto

```
ImageRD/
├── main.py                   # Punto de entrada CLI
├── ocr_engine.py             # Motor OCR con Tesseract
├── image_processing.py       # Preprocesamiento de imágenes
├── layout_reconstructor.py   # Reconstrucción de estructura visual
├── exporters.py              # Exportadores (TXT, JSON, DOCX)
├── utils.py                  # Utilidades y validaciones
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
```

---

## Formatos de salida

### Texto plano (`.txt`)
Texto limpio listo para copiar y pegar, con estructura preservada.

### JSON estructurado (`.json`)
```json
{
  "metadata": {
    "idioma": "spa",
    "total_palabras": 150,
    "confianza_promedio": 87.5
  },
  "texto_completo": "...",
  "bloques": [
    {
      "tipo": "texto",
      "coordenadas": [10, 20, 400, 50],
      "texto": "Contenido del bloque..."
    }
  ]
}
```

### Documento Word (`.docx`)
Documento formateado con párrafos, metadatos y estructura preservada.

---

## Ejemplo de salida en consola

```
═══════════════════════════════════════════════════════
  ImageRD — Sistema OCR Profesional v1.0.0
═══════════════════════════════════════════════════════

▸ Validando imagen...
  ✓ Imagen: documento.jpg
  ✓ Formato salida: docx
  ✓ Idioma: spa

▸ Procesando imagen...
  ✓ Palabras detectadas: 245
  ✓ Confianza promedio: 89.3%
  ✓ Bloques: 5

▸ Reconstruyendo estructura...
  ✓ Columnas detectadas: 1
  ✓ Multi-columna: No

▸ Generando archivos (docx)...

───────────────────────────────────────────────────────
  RESULTADO:
    ✓ documento_resultado.docx

  Tiempo total: 2.34s
───────────────────────────────────────────────────────
```

---

## Licencia

Proyecto de uso personal/educativo. Tesseract OCR está bajo licencia Apache 2.0.
