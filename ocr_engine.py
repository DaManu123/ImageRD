"""
ocr_engine.py — Motor OCR de alta precisión (v3).

Estrategia multi-paso optimizada:
  - Genera múltiples variantes de preprocesamiento
  - Prueba diferentes PSMs de Tesseract
  - Selecciona el resultado que maximiza CANTIDAD de palabras
    con confianza aceptable (no solo confianza alta)
  - Post-procesa texto: corrige comillas, caracteres confundidos
  - Fuerza DPI 300 en todas las llamadas a Tesseract
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from image_processing import (
    preprocess_image,
    generate_preprocessing_variants,
    smart_load,
    convert_to_grayscale,
    upscale_for_ocr,
    add_border_padding,
    resize_if_needed,
    BORDER_PADDING,
)
from utils import find_tesseract, logger, parse_languages


# ─── Configurar Tesseract ────────────────────
try:
    _tess = find_tesseract()
    pytesseract.pytesseract.tesseract_cmd = _tess
    logger.info("Tesseract: %s", _tess)
except EnvironmentError as err:
    logger.error(str(err))


# ─── Data classes ─────────────────────────────
@dataclass
class WordData:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    block_num: int
    par_num: int
    line_num: int
    word_num: int


@dataclass
class LineData:
    text: str
    words: List[WordData] = field(default_factory=list)
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    block_num: int = 0
    line_num: int = 0


@dataclass
class BlockData:
    block_num: int
    lines: List[LineData] = field(default_factory=list)
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    block_type: str = "texto"


@dataclass
class OCRResult:
    blocks: List[BlockData] = field(default_factory=list)
    raw_text: str = ""
    language: str = "spa"
    image_width: int = 0
    image_height: int = 0
    word_count: int = 0
    avg_confidence: float = 0.0


# ─── Motor OCR ────────────────────────────────
class OCREngine:
    """
    Motor OCR con estrategia multi-paso para máxima precisión.

    La selección del mejor resultado prioriza CANTIDAD de texto
    detectado (más palabras = más texto real capturado), con un
    umbral mínimo de confianza para filtrar basura.
    """

    # PSMs a probar: 3=auto, 6=bloque uniforme, 4=columna
    CANDIDATE_PSMS: List[int] = [3, 6, 4]

    def __init__(
        self,
        language: str = "spa",
        min_confidence: float = 25.0,
        psm: int = 3,
        multi_pass: bool = True,
    ) -> None:
        self.language = parse_languages(language)
        self.min_confidence = min_confidence
        self.psm = psm
        self.multi_pass = multi_pass

    def extract(self, image_path: str, preprocess: bool = True) -> OCRResult:
        """
        Extracción OCR completa.

        En modo multi-paso, prueba ~18 combinaciones (6 variantes × 3 PSMs)
        y elige la que captura más texto con confianza aceptable.
        """
        logger.info("═" * 50)
        logger.info("OCR v3 — Multi-paso: %s", "SÍ" if self.multi_pass else "NO")
        logger.info("Idioma: %s | Confianza mín: %.0f%%", self.language, self.min_confidence)

        if self.multi_pass and preprocess:
            result = self._multi_pass(image_path)
        else:
            result = self._single_pass(image_path, preprocess)

        # Post-procesamiento de texto
        result = self._post_process(result)

        logger.info("RESULTADO FINAL: %d palabras, %d bloques, conf: %.1f%%",
                     result.word_count, len(result.blocks), result.avg_confidence)
        logger.info("═" * 50)
        return result

    # ─── Multi-paso ───────────────────────────
    def _multi_pass(self, image_path: str) -> OCRResult:
        """
        Prueba múltiples combinaciones y elige la mejor.

        Score = word_count² × avg_confidence
        (Prioriza máxima cantidad de palabras con confianza decente)
        """
        variants = generate_preprocessing_variants(image_path)

        best: Optional[OCRResult] = None
        best_score = -1.0
        attempts = 0

        for vi, variant in enumerate(variants):
            h, w = variant.shape[:2]
            for psm in self.CANDIDATE_PSMS:
                attempts += 1
                try:
                    r = self._run_tesseract(variant, w, h, psm)
                    score = self._score(r)

                    if score > best_score:
                        best_score = score
                        best = r
                        logger.info(
                            "★ Mejor: V%d PSM%d — %d palabras, "
                            "conf %.1f%%, score %.0f",
                            vi, psm, r.word_count, r.avg_confidence, score,
                        )
                except Exception as e:
                    logger.debug("V%d PSM%d falló: %s", vi, psm, e)

        logger.info("Multi-paso: %d intentos, score final: %.0f", attempts, best_score)

        if best is None:
            return OCRResult(language=self.language)
        return best

    def _single_pass(self, image_path: str, preprocess: bool) -> OCRResult:
        """Extracción de un solo paso."""
        if preprocess:
            processed = preprocess_image(image_path)
        else:
            raw = smart_load(image_path)
            gray = convert_to_grayscale(raw)
            processed = upscale_for_ocr(gray)
            processed = add_border_padding(processed, BORDER_PADDING)

        h, w = processed.shape[:2]
        return self._run_tesseract(processed, w, h, self.psm)

    # ─── Tesseract ────────────────────────────
    def _run_tesseract(
        self, img: np.ndarray, w: int, h: int, psm: int,
    ) -> OCRResult:
        """Ejecuta Tesseract sobre imagen preprocesada."""
        config = f"--oem 3 --psm {psm} --dpi 300"

        try:
            data = pytesseract.image_to_data(
                img, lang=self.language, config=config,
                output_type=Output.DICT,
            )
        except pytesseract.TesseractError as exc:
            raise RuntimeError(
                f"Tesseract error: {exc}. ¿Idioma '{self.language}' instalado?"
            ) from exc

        words = self._parse_words(data)
        blocks = self._build_blocks(words)
        raw_text = self._build_raw_text(blocks)

        avg_conf = (sum(w.confidence for w in words) / len(words)) if words else 0.0

        return OCRResult(
            blocks=blocks, raw_text=raw_text, language=self.language,
            image_width=w, image_height=h,
            word_count=len(words), avg_confidence=avg_conf,
        )

    # ─── Scoring ──────────────────────────────
    @staticmethod
    def _score(r: OCRResult) -> float:
        """
        Score inteligente que balancea cantidad y calidad.

        1. Filtra palabras-artefacto (1 caracter no alfanumérico)
        2. Cuenta solo palabras "reales" (>=2 chars o alfanum comunes)
        3. Usa mediana de confianza (robusta vs outliers)
        4. Score = real_words × median_confidence
        """
        if r.word_count == 0:
            return 0.0

        # Contar palabras "reales" (no artefactos de UI)
        VALID_SINGLE = set("aeiouAEIOUyY0123456789OoXx=")
        real_words = 0
        confidences = []

        for block in r.blocks:
            for line in block.lines:
                for w in line.words:
                    # Aceptar palabras de 2+ chars, o chars útiles
                    if len(w.text) >= 2 or w.text in VALID_SINGLE:
                        real_words += 1
                        confidences.append(w.confidence)

        if not confidences:
            return 0.0

        # Mediana de confianza (no afectada por outliers)
        confidences.sort()
        median_conf = confidences[len(confidences) // 2]

        return real_words * median_conf

    # ─── Parsing ──────────────────────────────
    def _parse_words(self, data: Dict) -> List[WordData]:
        """Filtra palabras por confianza y limpia basura."""
        words: List[WordData] = []
        n = len(data["text"])

        for i in range(n):
            text = str(data["text"][i]).strip()
            conf = float(data["conf"][i])

            if not text or conf < self.min_confidence:
                continue

            # Filtrar caracteres basura aislados
            if len(text) == 1 and text in "|~`^":
                continue

            words.append(WordData(
                text=text, confidence=conf,
                x=int(data["left"][i]), y=int(data["top"][i]),
                width=int(data["width"][i]), height=int(data["height"][i]),
                block_num=int(data["block_num"][i]),
                par_num=int(data["par_num"][i]),
                line_num=int(data["line_num"][i]),
                word_num=int(data["word_num"][i]),
            ))

        return words

    # ─── Build blocks ─────────────────────────
    def _build_blocks(self, words: List[WordData]) -> List[BlockData]:
        """Agrupa palabras en bloques → líneas."""
        if not words:
            return []

        bmap: Dict[int, Dict[int, List[WordData]]] = {}
        for w in words:
            bn = w.block_num
            lk = w.par_num * 1000 + w.line_num
            bmap.setdefault(bn, {}).setdefault(lk, []).append(w)

        blocks: List[BlockData] = []
        for bn in sorted(bmap):
            lines: List[LineData] = []
            for lk in sorted(bmap[bn]):
                lw = sorted(bmap[bn][lk], key=lambda x: x.x)
                text = self._join_words(lw)
                lx = min(w.x for w in lw)
                ly = min(w.y for w in lw)
                lx2 = max(w.x + w.width for w in lw)
                ly2 = max(w.y + w.height for w in lw)
                lines.append(LineData(
                    text=text, words=lw,
                    x=lx, y=ly, width=lx2 - lx, height=ly2 - ly,
                    block_num=bn, line_num=lk,
                ))

            lines.sort(key=lambda l: l.y)
            if lines:
                bx = min(l.x for l in lines)
                by = min(l.y for l in lines)
                bx2 = max(l.x + l.width for l in lines)
                by2 = max(l.y + l.height for l in lines)
            else:
                bx = by = bx2 = by2 = 0

            blocks.append(BlockData(
                block_num=bn, lines=lines,
                x=bx, y=by, width=bx2 - bx, height=by2 - by,
            ))

        blocks.sort(key=lambda b: (b.y, b.x))
        return blocks

    @staticmethod
    def _join_words(words: List[WordData]) -> str:
        """Une palabras respetando espaciado real entre ellas."""
        if len(words) <= 1:
            return words[0].text if words else ""

        # Calcular gaps
        gaps = [
            words[i].x - (words[i - 1].x + words[i - 1].width)
            for i in range(1, len(words))
        ]

        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 1
        normal = max(median_gap, 1)

        parts = [words[0].text]
        for i, gap in enumerate(gaps):
            ratio = gap / normal if normal > 0 else 1
            parts.append("   " if ratio > 3.0 else " ")
            parts.append(words[i + 1].text)

        return "".join(parts)

    def _build_raw_text(self, blocks: List[BlockData]) -> str:
        parts = ["\n".join(l.text for l in b.lines) for b in blocks]
        return "\n\n".join(parts)

    # ─── Post-procesamiento ───────────────────
    def _post_process(self, result: OCRResult) -> OCRResult:
        """Corrige errores comunes de OCR en todo el resultado."""
        for block in result.blocks:
            for line in block.lines:
                line.text = _fix_ocr_text(line.text)
        result.raw_text = self._build_raw_text(result.blocks)
        return result

    def detect_image_regions(self, image_path: str) -> List[Dict]:
        """Detecta regiones con texto embebido en gráficos."""
        img = smart_load(image_path)
        gray = convert_to_grayscale(img) if len(img.shape) == 3 else img

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        min_area = gray.shape[0] * gray.shape[1] * 0.02
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < min_area:
                continue
            roi = gray[y:y + h, x:x + w]
            text = pytesseract.image_to_string(roi, lang=self.language, config="--psm 6").strip()
            if text:
                regions.append({
                    "tipo": "imagen_con_texto",
                    "coordenadas": [x, y, w, h],
                    "texto": text,
                    "descripcion": f"[Región ({x},{y}) — {w}x{h}px]",
                })
        return regions


# ─── Corrección de texto OCR ─────────────────
# Mapeo de caracteres tipográficos → ascii estándar
_CHAR_FIXES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"',   # "", „
    "\u00ab": '"', "\u00bb": '"',                    # «»
    "\u2018": "'", "\u2019": "'",                    # ''
    "\u2013": "-", "\u2014": "-", "\u2015": "-",     # –—―
    "\ufb01": "fi", "\ufb02": "fl",                  # ligaduras
    "\u00a0": " ",                                    # NBSP
}


def _fix_ocr_text(text: str) -> str:
    """
    Corrige errores frecuentes de OCR:
      - Comillas tipográficas → rectas
      - Ligaduras → caracteres separados
      - Espacios no rompibles → espacios normales
      - Caracteres confundidos en contexto de código
      - Espaciado alrededor de guiones bajos
    """
    import re

    # 1. Caracteres tipográficos → ASCII
    for old, new in _CHAR_FIXES.items():
        text = text.replace(old, new)

    # 2. ¡ → j en contexto de programación (error muy común en español)
    #    str(¡+1) → str(j+1),  =¡+1 → =j+1,  (¡) → (j), etc.
    text = re.sub(r'(?<=[(\[=+\-*/, ])¡(?=[+\-*/)0-9\]])', 'j', text)
    text = re.sub(r'\bstr\(¡', 'str(j', text)
    text = re.sub(r'=¡\+', '=j+', text)

    # 3. Separar palabras pegadas a guiones bajos: ________esun → ________ es un
    text = re.sub(r'(_+)([a-záéíóúA-ZÁÉÍÓÚ])', r'\1 \2', text)

    # 4. Colapsar espacios múltiples internos (>3) a uno solo
    #    pero preservar indentación al inicio de línea
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        # Preservar espacios de indentación al inicio
        stripped = line.lstrip(' ')
        indent = len(line) - len(stripped)
        # En el contenido, colapsar 3+ espacios a 1
        import re as _re
        stripped = _re.sub(r' {3,}', ' ', stripped)
        fixed_lines.append(' ' * indent + stripped)
    text = '\n'.join(fixed_lines)

    return text


# ─── Conveniencia ─────────────────────────────
def quick_extract(image_path: str, language: str = "spa", preprocess: bool = True) -> str:
    engine = OCREngine(language=language)
    return engine.extract(image_path, preprocess=preprocess).raw_text
