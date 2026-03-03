# ImageRD — Instrucciones para Agentes de IA

## Arquitectura

Pipeline lineal de 4 etapas con dos puntos de entrada (CLI y GUI):

```
main.py (CLI) / gui.py (GUI)
  → image_processing.py → ocr_engine.py → layout_reconstructor.py → exporters.py
```

Módulos transversales: `utils.py` (logging, validación, rutas, detección de Tesseract), `tesseract_manager.py` (descarga y gestión de Tesseract bundled en `vendor/tesseract/`).

**Flujo de datos:** `str` (ruta imagen) → `np.ndarray` (variantes preprocesadas) → `OCRResult` (dataclass) → `ReconstructedDocument` (texto con layout) → `Path` (archivos en `resultados/`).

**Nota:** `gui.py` NO llama a `run_pipeline()` — duplica los pasos del pipeline en `_ocr_worker()`. `layout_reconstructor.py` importa `_post_process_document()` desde `ocr_engine.py` (dependencia cruzada intencional).

## Convenciones del Proyecto

- **Bilingüe:** identificadores en inglés (`image_path`, `word_count`); strings de usuario, docstrings, comentarios, claves JSON y ayuda CLI en **español** (`"confianza_promedio"`, `"total_bloques"`)
- **Docstrings:** estilo Google con descripciones en español. Toda función pública lleva Args/Returns/Raises
- **Constantes:** `UPPER_SNAKE_CASE` (`BORDER_PADDING = 80`, `CANDIDATE_PSMS = [3, 6]`, `TARGET_MIN_DIMENSION = 2200`, `_EARLY_STOP_SCORE = 2400.0`)
- **Data classes:** `@dataclass` PascalCase — `WordData`, `LineData`, `BlockData`, `OCRResult` (en `ocr_engine.py`); `Column`, `ReconstructedDocument` (en `layout_reconstructor.py`)
- **Funciones privadas:** prefijo `_` (`_smart_upscale`, `_fix_ocr_text`)
- **Separadores de sección:** banners `# ─────────────────` en todos los archivos
- **Salida consola:** Unicode decorativo (`═`, `─`, `▸`, `✓`, `✗`, `│`, `○`, `☐`, `☑`)
- **Versión duplicada:** `__version__ = "1.0.0"` en `main.py` y `APP_VERSION = "1.0.0"` en `gui.py` — no compartida

## Decisiones Técnicas Críticas

1. **NO binarizar para screenshots.** Tesseract LSTM (OEM 3) rinde mejor con escala de grises. `binarize` en `preprocess_image()` es un no-op intencional (parámetro ghost)
2. **Upscaling agresivo (3x)** — principal factor de precisión OCR. Ver `_smart_upscale()` en `image_processing.py`
3. **`--dpi 300` siempre forzado** en toda llamada a Tesseract, independientemente del DPI real
4. **Scoring multi-paso:** `real_words × median_confidence` — favorece recall. Filtra artefactos de 1 carácter fuera de `VALID_SINGLE`
5. **Detección screenshot vs foto:** varianza Laplaciana (>300) + entropía Shannon (<6.8). Screenshots → procesamiento suave; fotos → procesamiento fuerte
6. **Confianza siempre auto-adaptativa:** `min_confidence` se fuerza a `0` internamente — `analyze_image_quality()` ajusta el umbral. `--min-confidence` existe como parámetro ghost (hidden en CLI)
7. **Multi-paso en fases con early stop (OCR v5):**
   - **Fase 1A:** Top 3 variantes × mejor PSM = 3 tareas (mínimo lanzamiento)
   - **Fase 1B:** Mismas variantes × PSMs restantes (solo si score Fase 1A < `_EARLY_STOP_SCORE`)
   - **Fase 2:** Variantes restantes (binarizadas/examen) × todos los PSMs (solo si Fase 1 insuficiente)
   - Early termination: al alcanzar score ≥ `_EARLY_STOP_SCORE`, cancela futures pendientes y omite fases siguientes
8. **Caching de imagen precargada:** `extract()` carga la imagen una vez y pasa `_preloaded` dict a todas las funciones de análisis/preprocesamiento, evitando recargas desde disco
9. **Solo PSMs 3 y 6:** PSM 4 (columna) eliminado — no aporta para screenshots ni fotos típicas

## Concurrencia

- **OCR multi-paso** (`ocr_engine.py`): `ThreadPoolExecutor` ejecuta combinaciones variante×PSM en paralelo con early termination. Tareas se envían por fases (1A→1B→2) para minimizar lanzamientos innecesarios. Tesseract corre como subproceso externo → no afectado por GIL. Workers auto-detectados vía `/proc/cpuinfo` en `get_optimal_workers()`
- **GUI** (`gui.py`): procesamiento OCR e instalación de Tesseract corren en `threading.Thread(daemon=True)`. Resultados al hilo principal vía `self.after(0, callback)`

## Flujo de Trabajo

```bash
python main.py --image documento.jpg --output txt --lang spa         # Básico
python main.py --image foto.png --output all --lang spa+eng          # Todos los formatos
python main.py --image captura.png --output txt --single-pass        # Sin multi-paso
python main.py --setup-tesseract                                     # Descarga Tesseract bundled
python main.py --add-lang fra                                        # Agrega idioma
python gui.py                                                        # Interfaz gráfica
```

Resultados en `resultados/{nombre}_{YYYY-MM-DD_HH-MM-SS}.{ext}`. Imágenes del portapapeles usan prefijo `imageRD_clip_`.

**Códigos de salida:** 0 = éxito, 1 = error (FileNotFoundError/ValueError/OSError/RuntimeError), 130 = Ctrl+C (convención Unix 128+SIGINT).

## Al Modificar Código

- **Nuevas variantes de preprocesamiento:** `generate_preprocessing_variants()` en `image_processing.py` — respetar exclusión de binarización para screenshots. Variantes ordenadas por prioridad (V0=full pipeline, V1=upscale puro, V2=CLAHE+sharpen). Aceptan `_preloaded` dict para evitar recargas
- **Nuevos formatos de exportación:** seguir patrón `export_to_txt/json/docx` y registrar en `export_all()`
- **Post-procesamiento OCR:** patrones en `_fix_ocr_text()`, correcciones globales en `_apply_global_fixes()`, detección de código en `_is_code_line()` — todo en `ocr_engine.py`
- **Parámetros ghost — NO eliminar:** `binarize` en `preprocess_image()`, `suffix` en `generate_output_path()`, `--min-confidence` en CLI (hidden con `argparse.SUPPRESS`)
- **GUI y CLI deben mantenerse sincronizados:** los cambios en opciones de `build_parser()` deben reflejarse en los widgets de `gui.py`
- **No hay test suite ni CI.** Validar cambios ejecutando: `python main.py --image <imagen_prueba> --output txt --verbose`
