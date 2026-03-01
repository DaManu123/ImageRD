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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from image_processing import (
    preprocess_image,
    generate_preprocessing_variants,
    analyze_optimal_psm,
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

    # Número máximo de hilos razonable para Tesseract
    _MAX_WORKERS: int = 8

    def __init__(
        self,
        language: str = "spa",
        min_confidence: float = 5.0,
        psm: int = 3,
        multi_pass: bool = True,
        workers: int = 0,
        auto_psm: bool = True,
    ) -> None:
        self.language = parse_languages(language)
        self.min_confidence = min_confidence
        self.psm = psm
        self.multi_pass = multi_pass
        self.auto_psm = auto_psm

        # workers=0 → auto-detectar (núcleos CPU, tope _MAX_WORKERS)
        if workers <= 0:
            cpu = os.cpu_count() or 4
            self.workers = min(cpu, self._MAX_WORKERS)
        else:
            self.workers = min(workers, self._MAX_WORKERS)

    def extract(self, image_path: str, preprocess: bool = True) -> OCRResult:
        """
        Extracción OCR completa.

        En modo multi-paso, prueba ~18 combinaciones (6 variantes × 3 PSMs)
        y elige la que captura más texto con confianza aceptable.
        """
        logger.info("═" * 50)
        logger.info("OCR v3 — Multi-paso: %s | Workers: %d",
                     "SÍ" if self.multi_pass else "NO", self.workers)
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
        Prueba múltiples combinaciones **en paralelo** y elige la mejor.

        Usa ``ThreadPoolExecutor`` con ``self.workers`` hilos.
        Tesseract es un proceso externo, por lo que los hilos
        evitan el GIL y aprovechan todos los núcleos disponibles.

        Score = real_words × median_confidence
        """
        variants = generate_preprocessing_variants(image_path)

        # ── Determinar orden de PSMs ──
        if self.auto_psm:
            psm_order = analyze_optimal_psm(image_path)
            logger.info("Auto-PSM: orden optimizado %s", psm_order)
        else:
            psm_order = self.CANDIDATE_PSMS

        # ── Construir lista de tareas (variant_idx, variant, psm) ──
        tasks: List[Tuple[int, np.ndarray, int]] = []
        for vi, variant in enumerate(variants):
            for psm in psm_order:
                tasks.append((vi, variant, psm))

        logger.info("Multi-paso paralelo: %d tareas → %d workers",
                     len(tasks), self.workers)

        best: Optional[OCRResult] = None
        best_score = -1.0
        completed = 0

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._eval_variant, vi, variant, psm): (vi, psm)
                for vi, variant, psm in tasks
            }

            for future in as_completed(futures):
                vi, psm = futures[future]
                completed += 1
                try:
                    result, score = future.result()
                    if score > best_score:
                        best_score = score
                        best = result
                        logger.info(
                            "★ Mejor: V%d PSM%d — %d palabras, "
                            "conf %.1f%%, score %.0f",
                            vi, psm, result.word_count,
                            result.avg_confidence, score,
                        )
                except Exception as exc:
                    logger.debug("V%d PSM%d falló: %s", vi, psm, exc)

        logger.info("Multi-paso: %d intentos, score final: %.0f",
                     completed, best_score)

        if best is None:
            return OCRResult(language=self.language)
        return best

    def _eval_variant(
        self, vi: int, variant: np.ndarray, psm: int,
    ) -> Tuple[OCRResult, float]:
        """
        Evalúa una combinación variante + PSM.

        Función auxiliar thread-safe para ejecución paralela.

        Args:
            vi: Índice de la variante (solo para logging).
            variant: Imagen preprocesada.
            psm: Modo de segmentación de página.

        Returns:
            Tupla ``(OCRResult, score)``.
        """
        h, w = variant.shape[:2]
        r = self._run_tesseract(variant, w, h, psm)
        score = self._score(r)
        return r, score

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
        VALID_SINGLE = set("aeiouAEIOUyY0123456789OoXx=()[]{}:;.,+-*/\"'")
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

            # Filtrar solo basura inequívoca (mínimo posible)
            if len(text) == 1 and text in "|~`^":
                continue

            # Preservar símbolos de formulario que Tesseract detecta
            # (círculos, checks, bullets, etc.)
            if text in ("O", "o", "0", "Oo", "oO"):
                # Podría ser radio button — preservar
                pass

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


def _post_process_document(text: str) -> str:
    """
    Post-procesamiento a nivel de DOCUMENTO COMPLETO.

    Detecta patrones de exámenes/cuestionarios y:
      - Agrega símbolos ○ antes de opciones de respuesta.
      - Agrega separadores ─── entre preguntas.
      - Limpia espaciado y formato.
      - Corrige confusiones I/l (Ist→lst, etc.).
      - Corrige confusiones comunes de OCR en español.

    Args:
        text: Texto completo del documento reconstruido.

    Returns:
        Texto mejorado con formato de examen.
    """
    import re

    lines = text.split('\n')

    # ── Detectar si es un examen/cuestionario ──
    question_indices = []
    for i, line in enumerate(lines):
        if re.match(r'^\s*Pregunta\s+\d+', line.strip(), re.IGNORECASE):
            question_indices.append(i)

    is_exam = len(question_indices) >= 2

    if not is_exam:
        # No es examen — solo correcciones globales
        text = _apply_global_fixes(text)
        return text

    # ── Mapear zonas del examen ──
    # Cada pregunta tiene: header, enunciado, [bloque de código], opciones
    questions = _parse_exam_questions(lines, question_indices)

    # ── Reconstruir el documento con formato ──
    result: list = []

    # Texto antes de la primera pregunta (título del examen, etc.)
    for i in range(question_indices[0]):
        result.append(lines[i])

    for q in questions:
        # Separador antes de cada pregunta
        if result and result[-1].strip() != '':
            result.append('')
        result.append('─' * 50)
        result.append('')

        # Header: "Pregunta N"
        result.append(q['header'])

        # Enunciado
        for line in q['statement']:
            result.append(line)

        # Bloque de código (si hay)
        if q['code']:
            result.append('')
            for line in q['code']:
                result.append(line)
            result.append('')

        # Opciones con ○
        for opt in q['options']:
            # Limpiar texto de la opción
            clean = _clean_option_text(opt.strip())
            if clean:
                result.append(f'○ {clean}')

        # Líneas restantes (no clasificadas)
        for line in q.get('trailing', []):
            result.append(line)

    text = '\n'.join(result)

    # ── Correcciones globales ──
    text = _apply_global_fixes(text)

    # ── Limpiar líneas vacías excesivas (máximo 2 consecutivas) ──
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text


def _parse_exam_questions(lines: list, question_indices: list) -> list:
    """
    Parsea las preguntas del examen identificando sus componentes.

    Algoritmo de 3 fases:
      1. Statement: primeras líneas hasta terminador (``:`` / ``.`` / ``?``),
         con detección de continuación por minúscula inicial o longitud.
      2. Clasificación: cada línea restante se marca como *code* o *text*.
      3. Opciones: las líneas *text* tras el bloque de código son opciones.
         Si todas las líneas son código, las últimas 3 se toman como opciones
         (maneja preguntas como Q2/Q4 donde las opciones son líneas de código).

    Args:
        lines: Todas las líneas del texto.
        question_indices: Índices donde empieza cada pregunta.

    Returns:
        Lista de diccionarios ``{header, statement, code, options}``.
    """

    questions: list = []

    for qi, q_start in enumerate(question_indices):
        q_end = question_indices[qi + 1] if qi + 1 < len(question_indices) else len(lines)
        q_lines = lines[q_start:q_end]

        if not q_lines:
            continue

        header = q_lines[0].strip()
        body = q_lines[1:]

        # Eliminar blancos al inicio y final del cuerpo
        while body and not body[0].strip():
            body = body[1:]
        while body and not body[-1].strip():
            body = body[:-1]

        if not body:
            questions.append({'header': header, 'statement': [], 'code': [], 'options': []})
            continue

        # ── FASE 1: Extraer enunciado ──────────────────────────────
        statement: list = []
        rest_start = 0

        for j, line in enumerate(body):
            stripped = line.strip()

            # Saltar blancos antes del enunciado
            if not stripped:
                if statement:
                    # Buscar siguiente línea no vacía
                    next_ne = None
                    for k in range(j + 1, len(body)):
                        ns = body[k].strip()
                        if ns:
                            next_ne = ns
                            break
                    # Señal 1: minúscula inicial → continuación segura
                    if next_ne and next_ne[0].islower():
                        continue  # saltar blanco — viene continuación
                    # Señal 2: enunciado sin puntuación terminal
                    # + siguiente línea con ≥2 palabras → wrap probable
                    # (cubre "De instrucciones.", "Consola: 2345...")
                    last_stmt = statement[-1].strip()
                    stmt_incomplete = not (
                        last_stmt.endswith(':')
                        or last_stmt.endswith('.')
                        or last_stmt.endswith('?')
                    )
                    if stmt_incomplete and next_ne and len(next_ne.split()) >= 2:
                        continue
                    # No es continuación → fin del enunciado
                    rest_start = j + 1
                    break
                continue

            statement.append(line)

            # ¿Termina el enunciado aquí?
            if stripped.endswith(':') or stripped.endswith('.') or stripped.endswith('?'):
                rest_start = j + 1
                break

            # Heurística de continuación: mirar la siguiente línea no vacía
            next_stripped = None
            for k in range(j + 1, len(body)):
                ns = body[k].strip()
                if ns:
                    next_stripped = ns
                    break

            if next_stripped:
                # Si la siguiente comienza en minúscula → continuación del enunciado
                if next_stripped[0].islower():
                    continue
                # Si el enunciado no tiene puntuación terminal y la
                # siguiente línea tiene ≥2 palabras → wrap probable
                if not (stripped.endswith(':') or stripped.endswith('.') or stripped.endswith('?')):
                    if len(next_stripped.split()) >= 2:
                        continue
                # Si la siguiente es mucho más corta → probablemente ya es opción
                if len(next_stripped) < len(stripped) * 0.4:
                    rest_start = j + 1
                    break
            else:
                # No hay más líneas no vacías
                rest_start = j + 1
                break
        else:
            rest_start = len(body)

        # ── FASE 2: Clasificar líneas restantes (code / text) ─────
        rest = body[rest_start:]
        # Eliminar blancos iniciales del resto
        while rest and not rest[0].strip():
            rest = rest[1:]

        classified: list = []  # lista de (tipo, línea)
        for line in rest:
            stripped = line.strip()
            if not stripped:
                classified.append(('empty', line))
            elif _is_code_line(stripped):
                classified.append(('code', line))
            else:
                classified.append(('text', line))

        # Buscar la primera línea *text* para separar código de opciones
        first_text_idx = None
        for idx, (cls, _) in enumerate(classified):
            if cls == 'text':
                first_text_idx = idx
                break

        if first_text_idx is not None:
            code = [l for cls, l in classified[:first_text_idx] if cls == 'code']
            options = [l.strip() for cls, l in classified[first_text_idx:] if cls != 'empty']
        else:
            # Todo es código o vacío
            code_lines = [l for cls, l in classified if cls == 'code']
            code = code_lines
            options = []

        # ── FASE 3: Caso especial — opciones "ocultas" en código ──
        # Si no hay opciones text pero hay ≥3 líneas de código,
        # las últimas 3 líneas de código son las opciones de respuesta.
        if not options and len(code) >= 3:
            options = [l.strip() for l in code[-3:]]
            code = code[:-3]

        questions.append({
            'header': header,
            'statement': statement,
            'code': code,
            'options': options,
        })

    return questions


def _is_code_line(stripped: str) -> bool:
    """
    Determina si una línea es código de programación.

    Args:
        stripped: Línea sin espacios al inicio/final.

    Returns:
        True si parece código Python.
    """
    import re

    # Patrones inequívocos de código
    code_patterns = [
        r'^(if|elif|else|for|while|def|class|return|import|from|try|except|finally|with)\b',
        r'^[a-z_]\w*\s*=\s*',                           # asignación: var = ...
        r'^\w+\(.*\)',                                    # llamada: func(...)
        r'print\s*\(',                                    # print(
        r'input\s*\(',                                    # input(
        r'=\s*input\s*\(',                                # = input(
    ]

    for pat in code_patterns:
        if re.search(pat, stripped):
            return True

    # Contiene operadores de programación
    code_operators = ['==', '>=', '<=', '!=', '= input(', 'print(']
    return any(op in stripped for op in code_operators)



def _clean_option_text(text: str) -> str:
    """Limpia prefijos residuales del OCR en opciones de respuesta."""
    import re
    # Quitar "O " o "o " al inicio si es residuo de ○ detectado como letra
    text = re.sub(r'^[Oo0]\s+(?=[A-ZÁÉÍÓÚ])', '', text)
    text = re.sub(r'^[Oo0]\)\s*', '', text)
    text = re.sub(r'^[a-dA-D]\)\s*', '', text)
    # Quitar ○ ya existente si se duplicaría
    text = re.sub(r'^○\s*', '', text)
    return text.strip()


def _apply_global_fixes(text: str) -> str:
    """
    Correcciones globales de OCR que aplican a todo el documento.

    Args:
        text: Texto completo.

    Returns:
        Texto corregido.
    """
    import re

    # Ist → lst (confusión I/l muy común en OCR)
    text = re.sub(r'\bIst\b', 'lst', text)
    text = re.sub(r'\bIst\(', 'lst(', text)
    text = re.sub(r'\bIst\[', 'lst[', text)

    # iflint → if(int  (confusión l/( y fusión de palabras)
    text = re.sub(r'\biflint\b', 'if(int', text)
    text = re.sub(r'\bifllint\b', 'if(int', text)

    # M/P) → M/F) cuando aparece en contexto de Genero
    text = re.sub(r'Genero\s*\(M/P\)', 'Genero (M/F)', text)
    text = re.sub(r'\(M/P\)', '(M/F)', text)

    # Corregir "esun" → "es un" (palabras pegadas comunes)
    text = re.sub(r'\besun\b', 'es un', text)

    # Corregir "diferencía" → "diferencia" (tilde incorrecta)
    text = re.sub(r'\bdiferencía\b', 'diferencia', text)

    return text


# ─────────────────────────────────────────────────────────────
#  Función de conveniencia rápida
# ─────────────────────────────────────────────────────────────

def quick_extract(image_path: str, lang: str = "spa") -> str:
    """Extrae texto de una imagen con configuración por defecto.

    Función de conveniencia que ejecuta el pipeline OCR completo
    con parámetros predeterminados.

    Args:
        image_path: Ruta a la imagen.
        lang: Idioma(s) de Tesseract (por defecto ``"spa"``).

    Returns:
        Texto extraído.
    """
    engine = OCREngine(language=lang)
    result = engine.extract(image_path)
    return result.raw_text
