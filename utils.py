"""
utils.py — Utilidades generales del proyecto OCR.

Contiene funciones auxiliares para validación de archivos,
detección de Tesseract, logging y configuración general.
"""

import os
import sys
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# ─────────────────────────────────────────────
# Formatos de imagen soportados
# ─────────────────────────────────────────────
SUPPORTED_FORMATS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"]

# ─────────────────────────────────────────────
# Configuración de logging
# ─────────────────────────────────────────────
def setup_logger(name: str = "ocr_project", level: int = logging.INFO) -> logging.Logger:
    """
    Configura y devuelve un logger con formato profesional.

    Args:
        name: Nombre del logger.
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Instancia de logging.Logger configurada.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()


# ─────────────────────────────────────────────
# Validación de archivos
# ─────────────────────────────────────────────
def validate_image_path(image_path: str) -> Path:
    """
    Valida que la ruta de imagen exista y tenga un formato soportado.

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Objeto Path validado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato no es soportado.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"El archivo no existe: {path}")

    if not path.is_file():
        raise ValueError(f"La ruta no apunta a un archivo: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Formato '{suffix}' no soportado. "
            f"Formatos válidos: {', '.join(SUPPORTED_FORMATS)}"
        )

    return path


def validate_output_format(output_format: str) -> str:
    """
    Valida el formato de salida solicitado.

    Args:
        output_format: Formato deseado (txt, docx, json, all).

    Returns:
        Formato validado en minúsculas.

    Raises:
        ValueError: Si el formato no es válido.
    """
    valid_formats = {"txt", "docx", "json", "all"}
    fmt = output_format.lower().strip()

    if fmt not in valid_formats:
        raise ValueError(
            f"Formato de salida '{fmt}' no válido. "
            f"Opciones: {', '.join(sorted(valid_formats))}"
        )

    return fmt


# ─────────────────────────────────────────────
# Detección de Tesseract OCR
# ─────────────────────────────────────────────
def find_tesseract() -> str:
    """
    Busca la instalación de Tesseract OCR en el sistema.

    Returns:
        Ruta al ejecutable de Tesseract.

    Raises:
        EnvironmentError: Si Tesseract no se encuentra instalado.
    """
    # Verificar si está en el PATH
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        return tesseract_path

    # Rutas comunes en Windows
    common_windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]

    for path in common_windows_paths:
        if os.path.isfile(path):
            return path

    # Rutas comunes en macOS / Linux
    unix_paths = [
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]

    for path in unix_paths:
        if os.path.isfile(path):
            return path

    raise EnvironmentError(
        "Tesseract OCR no encontrado.\n"
        "Instálalo según tu sistema operativo:\n"
        "  Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
        "  macOS   : brew install tesseract\n"
        "  Linux   : sudo apt install tesseract-ocr"
    )


# ─────────────────────────────────────────────
# Utilidades de rutas
# ─────────────────────────────────────────────
# Carpeta por defecto donde se guardan los resultados
RESULTS_DIR_NAME = "resultados"


def generate_output_path(
    input_path: Path,
    extension: str,
    output_dir: Optional[str] = None,
    suffix: str = "_resultado",
) -> Path:
    """
    Genera una ruta de salida dentro de la carpeta 'resultados/',
    con nombre basado en la imagen + fecha y hora actual.

    Formato: resultados/{nombre}_{YYYY-MM-DD_HH-MM-SS}.{ext}
    Ejemplo: resultados/descarga_2026-03-01_14-30-25.txt

    Args:
        input_path: Ruta de la imagen original.
        extension: Extensión del archivo de salida (sin punto).
        output_dir: Directorio de salida opcional (sobreescribe la carpeta por defecto).
        suffix: Sufijo (no usado actualmente, mantenido por compatibilidad).

    Returns:
        Ruta completa para el archivo de salida.
    """
    stem = input_path.stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{stem}_{timestamp}.{extension}"

    if output_dir:
        out = Path(output_dir)
    else:
        out = input_path.parent / RESULTS_DIR_NAME

    out.mkdir(parents=True, exist_ok=True)
    return out / filename


def parse_languages(lang_str: str) -> str:
    """
    Normaliza la cadena de idiomas para Tesseract.

    Args:
        lang_str: Idiomas separados por '+' (e.g., 'spa+eng').

    Returns:
        Cadena de idiomas normalizada.
    """
    parts = [p.strip() for p in lang_str.replace(",", "+").split("+") if p.strip()]
    if not parts:
        return "spa"
    return "+".join(parts)
