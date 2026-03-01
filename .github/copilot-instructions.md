# ImageRD — Instrucciones para Agentes de IA

## Arquitectura

Pipeline lineal de 4 etapas orquestado por CLI:

```
main.py → image_processing.py → ocr_engine.py → layout_reconstructor.py → exporters.py
```

`utils.py` es módulo transversal (logging, validación, rutas, detección de Tesseract).

**Flujo de datos:** `str` (ruta imagen) → `np.ndarray` (variantes preprocesadas) → `OCRResult` (dataclass con bloques/líneas/palabras) → `ReconstructedDocument` (texto con layout) → `Path` (archivos exportados).

## Convenciones del Proyecto

- **Bilingüe:** identificadores en inglés (`image_path`, `word_count`), strings de usuario, docstrings, comentarios, claves JSON y CLI en **español** (`"confianza_promedio"`, `"total_bloques"`)
- **Docstrings:** estilo Google con descripciones en español. Toda función pública lleva Args/Returns/Raises
- **Constantes:** `UPPER_SNAKE_CASE` (`BORDER_PADDING`, `CANDIDATE_PSMS`)
- **Data classes:** `@dataclass` PascalCase (`WordData`, `LineData`, `BlockData`, `OCRResult`, `Column`, `ReconstructedDocument`)
- **Funciones privadas:** prefijo `_` (`_smart_upscale`, `_fix_ocr_text`)
- **Separadores de sección:** banners con `# ─────────────────` en todos los archivos
- **Salida consola:** Unicode decorativo (`═`, `─`, `▸`, `✓`, `✗`, `│`) para UI profesional en terminal

## Decisiones Técnicas Críticas

1. **NO binarizar para screenshots.** Tesseract LSTM (OEM 3) rinde mejor con imágenes en escala de grises que binarizadas. El parámetro `binarize` en `preprocess_image()` es intencionalmente un no-op
2. **Upscaling agresivo (3x)** es el factor más importante para precisión OCR. Ver `_smart_upscale()` en `image_processing.py`
3. **`--dpi 300` siempre forzado** en cada llamada a Tesseract, independientemente del DPI real de la imagen
4. **Scoring multi-paso:** `real_words × median_confidence` — favorece recall sobre precisión. Filtra artefactos de un solo carácter que no estén en `VALID_SINGLE`
5. **Detección screenshot vs foto:** usa varianza Laplaciana (>300) + entropía Shannon (<6.8). Screenshots reciben procesamiento suave; fotos reciben procesamiento fuerte

## Estrategia Multi-Paso

`generate_preprocessing_variants()` genera 5 variantes (screenshots) o 7 (fotos):
- V0: pipeline completo, V1: solo sharpen, V2: solo CLAHE, V3: upscale puro (cero procesamiento), V4: CLAHE+sharpen
- V5-V6 (solo fotos): binarización Otsu y threshold adaptativo

Cada variante se prueba con PSMs [3, 6, 4] → hasta 21 combinaciones. La mejor se selecciona por score.

## Módulos Clave

| Archivo | Responsabilidad |
|---|---|
| `main.py` | CLI (argparse), pipeline `run_pipeline()`, manejo de excepciones con códigos de salida |
| `utils.py` | Logger singleton, validación, `generate_output_path()` (timestamps en `resultados/`), `find_tesseract()` multiplataforma |
| `image_processing.py` | Preprocesamiento CV2/Pillow; aliases públicos para cada función privada (compatibilidad hacia atrás) |
| `ocr_engine.py` | `OCREngine` con multi-paso, `_fix_ocr_text()` para errores comunes (¡→j en contexto código), `_join_words()` con gaps reales en píxeles |
| `layout_reconstructor.py` | Detección de columnas por cluster de centro-X, orden de lectura, indentación píxel→caracteres |
| `exporters.py` | Patrón dispatcher `export_all()` → TXT/JSON/DOCX. JSON con `ensure_ascii=False` y claves en español |

## Flujo de Trabajo

```bash
# Ejecución básica
python main.py --image documento.jpg --output txt --lang spa

# Todos los formatos
python main.py --image foto.png --output all --lang spa+eng

# Modo rápido (sin multi-paso)
python main.py --image captura.png --output txt --single-pass
```

Los resultados se guardan en `resultados/{nombre}_{YYYY-MM-DD_HH-MM-SS}.{ext}`.

## Al Modificar Código

- **Nuevas variantes de preprocesamiento:** agregarlas en `generate_preprocessing_variants()` de `image_processing.py`, respetando la convención de excluir binarización para screenshots
- **Nuevos formatos de exportación:** seguir el patrón de `export_to_txt/json/docx` y registrar en `export_all()`
- **Post-procesamiento OCR:** agregar patrones en `_fix_ocr_text()` de `ocr_engine.py`
- **Parámetros ghost:** `binarize` en `preprocess_image()` y `suffix` en `generate_output_path()` existen por compatibilidad — no eliminar
- No hay test suite ni CI. Validar cambios ejecutando el pipeline contra una imagen de prueba
