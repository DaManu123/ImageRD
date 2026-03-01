"""
main.py — Punto de entrada principal del sistema OCR ImageRD.

Proporciona una interfaz CLI profesional para:
  - Extraer texto de imágenes.
  - Reconstruir la estructura visual del documento.
  - Exportar resultados en múltiples formatos.

Uso:
    python main.py --image documento.jpg --output docx --lang spa
    python main.py --image foto.png --output all --lang spa+eng
    python main.py --image captura.webp --output json --verbose
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from utils import (
    logger,
    setup_logger,
    validate_image_path,
    validate_output_format,
)
from ocr_engine import OCREngine, OCRResult
from layout_reconstructor import (
    LayoutReconstructor,
    ReconstructedDocument,
    analyze_document_structure,
)
from exporters import export_all


# ─────────────────────────────────────────────
# Versión
# ─────────────────────────────────────────────
__version__ = "1.0.0"
APP_NAME = "ImageRD — Sistema OCR Profesional"


# ─────────────────────────────────────────────
# CLI con argparse
# ─────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """
    Construye el parser de argumentos de línea de comandos.

    Returns:
        ArgumentParser configurado con todos los argumentos soportados.
    """
    parser = argparse.ArgumentParser(
        prog="ImageRD",
        description=(
            f"{APP_NAME} v{__version__}\n"
            "Extrae texto de imágenes respetando la estructura visual original."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos de uso:\n"
            "  python main.py --image documento.jpg --output txt --lang spa\n"
            "  python main.py --image foto.png --output docx --lang eng\n"
            "  python main.py --image captura.webp --output all --lang spa+eng\n"
            "  python main.py --image scan.tiff --output json --verbose --no-preprocess\n"
        ),
    )

    # Argumentos obligatorios
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Ruta a la imagen de entrada (JPG, PNG, JPEG, WEBP, TIFF).",
    )

    # Formato de salida
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="txt",
        choices=["txt", "json", "docx", "all"],
        help="Formato de salida: txt, json, docx o all (default: txt).",
    )

    # Idioma
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="spa",
        help="Idioma(s) OCR separados por '+' (default: spa). Ej: spa+eng.",
    )

    # Ruta de salida personalizada
    parser.add_argument(
        "--output-path", "-op",
        type=str,
        default=None,
        help="Ruta personalizada para el archivo de salida.",
    )

    # Confianza mínima
    parser.add_argument(
        "--min-confidence", "-mc",
        type=float,
        default=5.0,
        help="Confianza mínima para incluir palabras (0-100, default: 5).",
    )

    # PSM (Page Segmentation Mode)
    parser.add_argument(
        "--psm",
        type=int,
        default=3,
        choices=[0, 1, 3, 4, 6, 7, 8, 11, 12, 13],
        help="Modo de segmentación de página de Tesseract (default: 3).",
    )

    # Control de preprocesamiento
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Desactivar preprocesamiento de imagen.",
    )

    # Multi-paso
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Desactivar estrategia multi-paso (más rápido, menos preciso).",
    )

    # Workers (paralelismo)
    parser.add_argument(
        "--workers", "-w",
        type=int, default=0,
        help=(
            "Número de hilos para procesamiento paralelo. "
            "0 = auto-detectar según núcleos CPU (por defecto)."
        ),
    )

    # Verbosidad
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Activar modo detallado (debug).",
    )

    # Versión
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────
def run_pipeline(
    image_path: str,
    output_format: str = "txt",
    language: str = "spa",
    output_path: Optional[str] = None,
    min_confidence: float = 5.0,
    psm: int = 3,
    preprocess: bool = True,
    multi_pass: bool = True,
    workers: int = 0,
) -> List[Path]:
    """
    Ejecuta el pipeline completo de extracción OCR.

    Pasos:
      1. Validar imagen de entrada.
      2. Ejecutar motor OCR con preprocesamiento.
      3. Reconstruir estructura del documento.
      4. Exportar resultados.

    Args:
        image_path: Ruta a la imagen.
        output_format: Formato de salida.
        language: Idioma(s) de OCR.
        output_path: Ruta de salida personalizada.
        min_confidence: Umbral mínimo de confianza.
        psm: Modo de segmentación de página.
        preprocess: Aplicar preprocesamiento.

    Returns:
        Lista de archivos generados.

    Raises:
        FileNotFoundError: Si la imagen no existe.
        ValueError: Si los parámetros no son válidos.
        RuntimeError: Si el OCR falla.
    """
    start_time = time.time()

    # ── Paso 1: Validación ──
    print(f"\n{'═' * 55}")
    print(f"  {APP_NAME} v{__version__}")
    print(f"{'═' * 55}\n")

    print("▸ Validando imagen...")
    validated_path = validate_image_path(image_path)
    validated_format = validate_output_format(output_format)
    print(f"  ✓ Imagen: {validated_path.name}")
    print(f"  ✓ Formato salida: {validated_format}")
    print(f"  ✓ Idioma: {language}")
    print(f"  ✓ Multi-paso: {'Sí' if multi_pass else 'No'}")

    # ── Paso 2: Extracción OCR ──
    print("\n▸ Procesando imagen (esto puede tomar unos segundos)...")
    engine = OCREngine(
        language=language,
        min_confidence=min_confidence,
        psm=psm,
        multi_pass=multi_pass,
        workers=workers,
        auto_psm=True,
    )
    ocr_result: OCRResult = engine.extract(
        str(validated_path),
        preprocess=preprocess,
    )

    if ocr_result.word_count == 0:
        print("\n  ⚠ No se detectó texto en la imagen.")
        print("  Sugerencias:")
        print("    - Verifica que la imagen contenga texto legible.")
        print("    - Prueba con --no-preprocess si el preprocesamiento distorsiona.")
        print("    - Ajusta --min-confidence a un valor menor.")
        print("    - Verifica que el idioma (--lang) sea correcto.")
        return []

    print(f"  ✓ Palabras detectadas: {ocr_result.word_count}")
    print(f"  ✓ Confianza promedio: {ocr_result.avg_confidence:.1f}%")
    print(f"  ✓ Bloques: {len(ocr_result.blocks)}")

    # ── Paso 3: Reconstrucción de estructura ──
    print("\n▸ Reconstruyendo estructura...")
    reconstructor = LayoutReconstructor()
    reconstructed: ReconstructedDocument = reconstructor.reconstruct(ocr_result)
    print(f"  ✓ Columnas detectadas: {reconstructed.column_count}")
    print(f"  ✓ Multi-columna: {'Sí' if reconstructed.is_multicolumn else 'No'}")

    # ── Paso 4: Exportación ──
    print(f"\n▸ Generando archivos ({validated_format})...")
    generated_files = export_all(
        ocr_result=ocr_result,
        reconstructed=reconstructed,
        output_format=validated_format,
        output_path=output_path,
        input_path=str(validated_path),
    )

    # ── Resumen ──
    elapsed = time.time() - start_time

    print(f"\n{'─' * 55}")
    print("  RESULTADO:")
    for file_path in generated_files:
        print(f"    ✓ {file_path}")
    print(f"\n  Tiempo total: {elapsed:.2f}s")
    print(f"{'─' * 55}\n")

    # Mostrar vista previa del texto
    _show_preview(reconstructed.formatted_text)

    return generated_files


def _show_preview(text: str, max_lines: int = 15) -> None:
    """
    Muestra una vista previa del texto extraído en consola.

    Args:
        text: Texto completo.
        max_lines: Máximo de líneas a mostrar.
    """
    lines = text.split("\n")

    print("  VISTA PREVIA DEL TEXTO:")
    print(f"  {'·' * 45}")

    for line in lines[:max_lines]:
        print(f"  │ {line}")

    if len(lines) > max_lines:
        print(f"  │ ... ({len(lines) - max_lines} líneas más)")

    print(f"  {'·' * 45}\n")


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────
def main() -> None:
    """Punto de entrada principal del programa."""
    parser = build_parser()
    args = parser.parse_args()

    # Configurar logging verbose
    if args.verbose:
        import logging
        setup_logger(level=logging.DEBUG)

    try:
        run_pipeline(
            image_path=args.image,
            output_format=args.output,
            language=args.lang,
            output_path=args.output_path,
            min_confidence=args.min_confidence,
            psm=args.psm,
            preprocess=not args.no_preprocess,
            multi_pass=not args.single_pass,
            workers=args.workers,
        )
    except FileNotFoundError as exc:
        print(f"\n  ✗ Error: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n  ✗ Error de configuración: {exc}")
        sys.exit(1)
    except OSError as exc:
        print(f"\n  ✗ Error de entorno/archivo: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n  ✗ Error en OCR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  Operación cancelada por el usuario.")
        sys.exit(130)


if __name__ == "__main__":
    main()
