"""
image_processing.py — Preprocesamiento optimizado para Tesseract LSTM (v3).

Principio clave: el motor LSTM de Tesseract (OEM 3) funciona MEJOR
con imágenes en escala de grises de alta resolución que con imágenes
binarizadas. La binarización destruye información que el LSTM necesita.

Pipeline optimizado:
  1. Carga inteligente (OpenCV + Pillow fallback)
  2. Escala de grises
  3. Upscaling agresivo (2x-3x mínimo) — FACTOR MÁS IMPORTANTE
  4. Corrección de inclinación
  5. Mejora suave de contraste (CLAHE con parámetros bajos)
  6. Sharpening para definir bordes de caracteres
  7. Padding blanco en bordes
  8. SIN binarización para modo LSTM (variantes opcionales)
"""

import math
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from utils import logger


# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────
BORDER_PADDING = 80          # px — margen blanco (mayor para evitar recorte)
MIN_UPSCALE_FACTOR = 2.0     # Siempre escalar al menos 2x
MAX_UPSCALE_FACTOR = 4.0     # No escalar más de 4x


# ─────────────────────────────────────────────
# Pipeline principal (optimizado para LSTM)
# ─────────────────────────────────────────────
def preprocess_image(
    image_path: str,
    enhance_contrast: bool = True,
    denoise: bool = True,
    deskew: bool = True,
    binarize: bool = True,  # Se ignora para LSTM — mantenido por compatibilidad
) -> np.ndarray:
    """
    Pipeline de preprocesamiento optimizado para Tesseract LSTM.

    IMPORTANTE: No binariza la imagen a menos que se detecte
    que es una foto/escaneo de baja calidad. El LSTM de Tesseract
    obtiene mejor precisión con imágenes en escala de grises.

    Args:
        image_path: Ruta al archivo de imagen.
        enhance_contrast: Mejora de contraste (CLAHE suave).
        denoise: Reducción de ruido.
        deskew: Corrección de inclinación.
        binarize: Ignorado para LSTM (por diseño).

    Returns:
        Imagen preprocesada (escala de grises, alta resolución).
    """
    logger.info("Cargando imagen: %s", image_path)
    img = smart_load(image_path)

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    logger.info("Imagen: %dx%d, %d canales", w, h, channels)

    # Detectar tipo
    is_screenshot = _detect_if_screenshot(img)
    logger.info("Tipo: %s", "SCREENSHOT" if is_screenshot else "FOTO/ESCANEO")

    # 1. Escala de grises
    gray = _to_gray(img)

    # 2. Upscaling — EL paso más importante para precisión
    gray = _smart_upscale(gray)

    # 3. Deskew
    if deskew:
        gray = _deskew(gray)

    # 4. Procesamiento según tipo
    if is_screenshot:
        processed = _process_screenshot(gray, enhance_contrast)
    else:
        processed = _process_photo(gray, enhance_contrast, denoise)

    # 5. Padding
    processed = _add_padding(processed)

    logger.info("Preprocesamiento OK: %dx%d",
                processed.shape[1], processed.shape[0])
    return processed


def generate_preprocessing_variants(image_path: str) -> List[np.ndarray]:
    """
    Genera variantes para OCR multi-paso.

    Para SCREENSHOTS: solo variantes en escala de grises (LSTM-optimizado).
    Para FOTOS: incluye variantes con binarización.

    Returns:
        Lista de imágenes numpy array.
    """
    logger.info("Generando variantes de preprocesamiento...")
    img = smart_load(image_path)
    is_ss = _detect_if_screenshot(img)
    gray = _to_gray(img)
    up = _smart_upscale(gray)

    variants: List[np.ndarray] = []

    # V0: Pipeline estándar completo
    v0 = preprocess_image(image_path)
    variants.append(v0)

    # V1: Upscale + sharpen suave (mínimo procesamiento)
    v1 = _sharpen_gentle(up.copy())
    variants.append(_add_padding(v1))

    # V2: Upscale + CLAHE suave
    v2 = _clahe(up.copy(), clip=1.2, tile=(16, 16))
    variants.append(_add_padding(v2))

    # V3: Upscale puro (CERO procesamiento) — sorprendentemente bueno
    variants.append(_add_padding(up.copy()))

    # V4: CLAHE + sharpen combinados
    v4 = _clahe(up.copy(), clip=1.0, tile=(16, 16))
    v4 = _sharpen_gentle(v4)
    variants.append(_add_padding(v4))

    if not is_ss:
        # Solo para fotos/escaneos: incluir binarización
        # El LSTM funciona PEOR con binarización en screenshots
        logger.info("Foto detectada: añadiendo variantes binarizadas...")

        # V5: Otsu
        v5 = _clahe(up.copy(), clip=1.5)
        _, v5 = cv2.threshold(v5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(_add_padding(v5))

        # V6: Adaptativa suave
        v6 = _clahe(up.copy(), clip=1.5)
        v6 = cv2.adaptiveThreshold(
            v6, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
        variants.append(_add_padding(v6))

    logger.info("Variantes: %d (%s)", len(variants),
                "screenshot-opt" if is_ss else "foto-opt")
    return variants


# ─────────────────────────────────────────────
# Procesamiento por tipo
# ─────────────────────────────────────────────
def _process_screenshot(
    gray: np.ndarray,
    enhance_contrast: bool,
) -> np.ndarray:
    """
    Pipeline para screenshots — procesamiento MÍNIMO.

    Screenshots ya tienen texto limpio y nítido.
    Solo necesitan: upscale + contraste suave + sharpen suave.
    NO binarizar.
    """
    logger.info("Pipeline screenshot (suave)...")
    result = gray.copy()

    if enhance_contrast:
        result = _clahe(result, clip=1.2, tile=(16, 16))

    result = _sharpen_gentle(result)

    return result


def _process_photo(
    gray: np.ndarray,
    enhance_contrast: bool,
    denoise: bool,
) -> np.ndarray:
    """
    Pipeline para fotos/escaneos — procesamiento moderado.

    Fotos tienen ruido, iluminación desigual, etc.
    Necesitan: contraste + denoising + sharpen.
    SIN binarización (LSTM maneja bien escala de grises).
    """
    logger.info("Pipeline foto/escaneo (moderado)...")
    result = gray.copy()

    if enhance_contrast:
        result = _clahe(result, clip=2.0, tile=(8, 8))

    if denoise:
        result = _denoise(result, strength=6)

    result = _sharpen_strong(result)

    return result


# ─────────────────────────────────────────────
# Detección de tipo
# ─────────────────────────────────────────────
def _detect_if_screenshot(image: np.ndarray) -> bool:
    """
    Detecta si la imagen es screenshot vs foto.

    Screenshots: bordes nítidos, colores uniformes, baja entropía.
    Fotos: ruido, variación continua, alta entropía.
    """
    gray = _to_gray(image) if len(image.shape) == 3 else image

    # Reducir para análisis rápido
    small = cv2.resize(gray, (300, 300), interpolation=cv2.INTER_AREA)

    laplacian_var = cv2.Laplacian(small, cv2.CV_64F).var()

    hist = cv2.calcHist([small], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    nz = hist[hist > 0]
    entropy = -np.sum(nz * np.log2(nz))

    # Screenshot: alta nitidez + baja entropía
    is_ss = laplacian_var > 300 and entropy < 6.8

    logger.debug("Detección — Laplacian: %.0f, Entropy: %.2f → %s",
                 laplacian_var, entropy, "SS" if is_ss else "FOTO")
    return is_ss


# ─────────────────────────────────────────────
# Escala de grises
# ─────────────────────────────────────────────
def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Alias de compatibilidad
convert_to_grayscale = _to_gray


# ─────────────────────────────────────────────
# Upscaling inteligente
# ─────────────────────────────────────────────
def _smart_upscale(gray: np.ndarray) -> np.ndarray:
    """
    Upscaling agresivo para máxima precisión OCR.

    Tesseract necesita al menos ~300 DPI. La mayoría de screenshots
    son ~96 DPI, así que 3x es el mínimo ideal.

    Caracteres finos (comillas, paréntesis, puntos) necesitan
    resolución alta para ser reconocidos correctamente.
    """
    h, w = gray.shape[:2]

    # Para imágenes de menos de 10 megapíxeles: escalar 3x
    # Para imágenes grandes: 2x
    # Para imágenes enormes: 1.5x
    total_px = h * w
    if total_px > 20_000_000:  # >20 MP
        scale = 1.5
    elif total_px > 8_000_000:  # >8 MP
        scale = 2.0
    else:
        scale = 3.0

    # Nunca superar 4x
    scale = min(scale, MAX_UPSCALE_FACTOR)
    # Mínimo 2x siempre
    scale = max(scale, MIN_UPSCALE_FACTOR)

    new_h = int(h * scale)
    new_w = int(w * scale)

    logger.info("Upscale: x%.1f (%dx%d → %dx%d)", scale, w, h, new_w, new_h)

    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


# Alias de compatibilidad
def upscale_for_ocr(gray: np.ndarray, target_height: int = 2000) -> np.ndarray:
    """Alias de compatibilidad."""
    return _smart_upscale(gray)


# ─────────────────────────────────────────────
# CLAHE (contraste)
# ─────────────────────────────────────────────
def _clahe(
    gray: np.ndarray,
    clip: float = 1.5,
    tile: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE con parámetros configurables."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(gray)


# Alias
def apply_contrast_enhancement(
    gray: np.ndarray,
    clip_limit: float = 1.5,
    tile_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    return _clahe(gray, clip_limit, tile_size)


# ─────────────────────────────────────────────
# Denoising
# ─────────────────────────────────────────────
def _denoise(gray: np.ndarray, strength: int = 6) -> np.ndarray:
    """Non-Local Means Denoising."""
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def apply_denoising(gray: np.ndarray, strength: int = 6) -> np.ndarray:
    return _denoise(gray, strength)


# ─────────────────────────────────────────────
# Sharpening
# ─────────────────────────────────────────────
def _sharpen_gentle(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask suave — refuerza bordes sin amplificar ruido."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)


def _sharpen_strong(gray: np.ndarray) -> np.ndarray:
    """Kernel de sharpening fuerte para imágenes borrosas."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)


def apply_gentle_sharpen(gray: np.ndarray) -> np.ndarray:
    return _sharpen_gentle(gray)


def apply_strong_sharpen(gray: np.ndarray) -> np.ndarray:
    return _sharpen_strong(gray)


# ─────────────────────────────────────────────
# Binarización (solo para variantes, no pipeline principal)
# ─────────────────────────────────────────────
def apply_otsu_binarization(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_adaptive_binarization(
    gray: np.ndarray,
    block_size: int = 21,
    constant: int = 7,
) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, constant,
    )


# ─────────────────────────────────────────────
# Deskew
# ─────────────────────────────────────────────
def _detect_angle(gray: np.ndarray) -> float:
    """Detecta ángulo de inclinación con Hough."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100,
        minLineLength=gray.shape[1] // 8, maxLineGap=10,
    )
    if lines is None:
        return 0.0

    angles = [
        math.degrees(math.atan2(y2 - y1, x2 - x1))
        for [[x1, y1, x2, y2]] in lines
        if abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) < 15
    ]
    return float(np.median(angles)) if angles else 0.0


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Corrige inclinación."""
    angle = _detect_angle(gray)
    if abs(angle) < 0.3 or abs(angle) > 10:
        return gray
    logger.info("Deskew: %.2f°", angle)
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ─────────────────────────────────────────────
# Padding
# ─────────────────────────────────────────────
def _add_padding(gray: np.ndarray, pad: int = BORDER_PADDING) -> np.ndarray:
    """Agrega borde blanco."""
    return cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


def add_border_padding(gray: np.ndarray, padding: int = BORDER_PADDING) -> np.ndarray:
    return _add_padding(gray, padding)


# ─────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────
def load_image_pillow(image_path: str) -> np.ndarray:
    """Carga con Pillow (WEBP, formatos exóticos)."""
    try:
        pil = Image.open(image_path).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise IOError(f"No se pudo cargar: {exc}") from exc


def smart_load(image_path: str) -> np.ndarray:
    """Carga con OpenCV, fallback a Pillow."""
    img = cv2.imread(image_path)
    if img is not None:
        return img
    logger.warning("OpenCV falló, usando Pillow...")
    return load_image_pillow(image_path)


def resize_if_needed(image: np.ndarray, min_height: int = 600, max_height: int = 8000) -> np.ndarray:
    """Compatibilidad — ya no se usa en pipeline principal."""
    return image


# ─────────────────────────────────────────────
# Auto-detección de PSM óptimo
# ─────────────────────────────────────────────
def analyze_optimal_psm(image_path: str) -> List[int]:
    """
    Analiza la estructura visual de la imagen para ordenar los PSMs
    de más a menos probable.

    Análisis:
      - Aspecto ratio: imágenes altas y estrechas → PSM 4 (columna).
      - Densidad de texto: mucho texto → PSM 6 (bloque uniforme).
      - Líneas horizontales: separadores → PSM 3 (auto).
      - Texto disperso: pocos bloques → PSM 11 (sparse).

    Args:
        image_path: Ruta a la imagen.

    Returns:
        Lista de PSMs ordenados de mejor a peor.
    """
    img = smart_load(image_path)
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    aspect = h / max(w, 1)

    # Reducción para análisis rápido
    small = cv2.resize(gray, (min(w, 600), min(h, 600)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    # 1. Detectar líneas horizontales (bordes/separadores)
    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=sw // 3, maxLineGap=10)
    h_lines = 0
    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 5:  # casi horizontal
                h_lines += 1

    # 2. Densidad de píxeles oscuros (texto)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_density = np.sum(bw > 0) / (sh * sw)

    # 3. Contornos para contar bloques de texto
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filtrar contornos pequeños
    big_contours = [c for c in contours if cv2.contourArea(c) > (sh * sw * 0.0005)]
    n_blocks = len(big_contours)

    logger.info("Análisis PSM — aspect: %.2f, h_lines: %d, density: %.3f, blocks: %d",
                 aspect, h_lines, text_density, n_blocks)

    # Clasificación basada en características
    scores = {3: 0, 6: 0, 4: 0}

    # Imagen muy alta (exámenes, formularios) → PSM 6 es ideal
    if aspect > 2.0:
        scores[6] += 3
        scores[4] += 1
    elif aspect > 1.0:
        scores[6] += 2
    else:
        scores[3] += 2  # Imágenes anchas: auto funciona bien

    # Muchos separadores horizontales (exámenes)
    if h_lines >= 5:
        scores[6] += 2
        scores[3] += 1

    # Alta densidad de texto
    if text_density > 0.15:
        scores[6] += 2
    elif text_density > 0.08:
        scores[6] += 1
        scores[3] += 1

    # Muchos bloques de texto
    if n_blocks > 20:
        scores[6] += 1
    elif n_blocks < 5:
        scores[3] += 1

    # Ordenar PSMs por score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = [psm for psm, _ in ranked]

    logger.info("PSM ranking: %s", result)
    return result
