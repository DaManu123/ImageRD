"""ocr_engine.py - Motor OCR de alta precision (v4).

Estrategia multi-paso optimizada:
  - Genera múltiples variantes de preprocesamiento
  - Prueba diferentes PSMs de Tesseract
  - Selecciona el resultado que maximiza CANTIDAD de palabras
    con confianza aceptable (no solo confianza alta)
  - Post-procesa texto: corrige comillas, caracteres confundidos
  - Normaliza radio buttons (○) y checkboxes (☐) de formularios web
  - Limpia artefactos de OCR con regex especializados para exámenes
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
    analyze_image_quality,
    smart_load,
    convert_to_grayscale,
    upscale_for_ocr,
    add_border_padding,
    resize_if_needed,
    BORDER_PADDING,
)
from utils import find_tesseract, logger, parse_languages


# ─── Configurar Tesseract ────────────────────
_tesseract_ready = False

try:
    _tess = find_tesseract()
    pytesseract.pytesseract.tesseract_cmd = _tess
    logger.info("Tesseract: %s", _tess)
    _tesseract_ready = True
except EnvironmentError:
    # Tesseract no encontrado — se intentará auto-setup al primer uso
    logger.warning("Tesseract no encontrado. Se ofrecerá descarga automática al procesar.")


def _ensure_tesseract() -> None:
    """
    Asegura que Tesseract esté configurado antes de procesar.

    Si Tesseract no fue encontrado al importar el módulo, intenta
    descargarlo automáticamente usando tesseract_manager.
    """
    global _tesseract_ready

    if _tesseract_ready:
        return

    try:
        from tesseract_manager import ensure_tesseract as _ensure

        tess_path = _ensure(interactive=True)
        if tess_path:
            pytesseract.pytesseract.tesseract_cmd = tess_path
            logger.info("Tesseract configurado: %s", tess_path)
            _tesseract_ready = True
        else:
            raise EnvironmentError(
                "Tesseract OCR no disponible.\n"
                "Ejecuta: python tesseract_manager.py setup"
            )
    except ImportError:
        raise EnvironmentError(
            "Tesseract OCR no encontrado y tesseract_manager no disponible.\n"
            "Instala Tesseract manualmente o ejecuta: python tesseract_manager.py setup"
        )


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


# ─── Detección de CPU y workers óptimos ───────
def get_cpu_info() -> Dict[str, int]:
    """Detecta núcleos físicos e hilos lógicos del CPU.

    Intenta leer la topología real (cores × threads-per-core).
    Fallback a ``os.cpu_count()`` si no puede determinar la topología.

    Returns:
        Diccionario con claves ``'cores'``, ``'threads'``, ``'sockets'``.
    """
    threads = os.cpu_count() or 4
    cores = threads
    sockets = 1

    # Intentar leer topología real en Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        # Contar núcleos físicos únicos (core id por procesador)
        physical_ids = set()
        core_ids = set()
        current_phys = "0"
        for line in cpuinfo.splitlines():
            if line.startswith("physical id"):
                current_phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":")[1].strip()
                core_ids.add((current_phys, core_id))
                physical_ids.add(current_phys)
        if core_ids:
            cores = len(core_ids)
            sockets = len(physical_ids)
    except (OSError, ValueError):
        # Fallback: asumir hyper-threading 2:1 si threads > 1
        if threads > 1:
            cores = max(1, threads // 2)

    return {"cores": cores, "threads": threads, "sockets": sockets}


def get_optimal_workers(task_count: int = 0) -> int:
    """Calcula el número óptimo de workers paralelos.

    Tesseract se ejecuta como proceso externo (subprocess), por lo que
    los threads de Python NO compiten por el GIL. Podemos usar hasta
    todos los hilos lógicos del CPU.

    La fórmula reserva 1-2 hilos para el sistema y la GUI, y no
    crea más workers que tareas pendientes.

    Args:
        task_count: Número de tareas a ejecutar (0 = sin límite).

    Returns:
        Número de workers óptimo (≥ 1).
    """
    info = get_cpu_info()
    threads = info["threads"]

    # Reservar hilos para el sistema / GUI:
    # ≤4 hilos: reservar 1, ≤8: reservar 2, >8: reservar 2
    if threads <= 4:
        reserved = 1
    else:
        reserved = 2

    optimal = max(1, threads - reserved)

    # No crear más workers que tareas
    if task_count > 0:
        optimal = min(optimal, task_count)

    return optimal


# ─── Motor OCR ────────────────────────────────
class OCREngine:
    """
    Motor OCR con estrategia multi-paso para máxima precisión.

    La selección del mejor resultado prioriza CANTIDAD de texto
    detectado (más palabras = más texto real capturado), con un
    umbral mínimo de confianza para filtrar basura.
    """

    # PSMs a probar: 3=auto, 6=bloque uniforme (PSM 4 eliminado — rara vez gana)
    CANDIDATE_PSMS: List[int] = [3, 6]

    # Umbral de score para early termination en Fase 1.
    # Si una combinación supera este score, no se prueban más variantes.
    # Calculado como: ~30 palabras reales × ~80% confianza mediana = 2400
    _EARLY_STOP_SCORE: float = 2400.0

    def __init__(
        self,
        language: str = "spa",
        min_confidence: float = 0.0,
        psm: int = 3,
        multi_pass: bool = True,
        workers: int = 0,
        auto_psm: bool = True,
    ) -> None:
        self.language = parse_languages(language)
        # Confianza siempre auto-adaptativa (ignora valores del usuario)
        self.min_confidence = 0.0
        self.psm = psm
        self.multi_pass = multi_pass
        self.auto_psm = auto_psm

        # workers=0 → auto-detectar: usar todos los hilos lógicos del CPU.
        # Tesseract es proceso externo (subprocess) → los threads de Python
        # no compiten por el GIL, así que podemos usar todos los hilos.
        if workers <= 0:
            self.workers = get_optimal_workers()
        else:
            self.workers = max(1, workers)

    def extract(self, image_path: str, preprocess: bool = True) -> OCRResult:
        """
        Extracción OCR completa.

        En modo multi-paso, usa estrategia de dos fases:
          - Fase 1: variantes prioritarias × PSMs óptimos (rápido, ~4 llamadas).
          - Fase 2: variantes restantes (solo si Fase 1 no fue suficiente).

        La confianza es siempre auto-adaptativa según calidad de imagen.
        """
        # Asegurar que Tesseract esté disponible
        _ensure_tesseract()

        # ── Cargar imagen una sola vez y compartir entre análisis ──
        img = smart_load(image_path)
        gray = convert_to_grayscale(img)
        _preloaded = {"img": img, "gray": gray}

        # ── Confianza siempre adaptativa ──
        quality = analyze_image_quality(image_path, _preloaded=_preloaded)
        self.min_confidence = quality["suggested_confidence"]
        logger.info("Confianza adaptativa: calidad=%s → umbral=%.0f%%",
                     quality["quality_tier"], self.min_confidence)

        cpu_info = get_cpu_info()
        logger.info("═" * 50)
        logger.info("OCR v5 — Multi-paso: %s | Workers: %d/%d hilos (%d cores)",
                     "SÍ" if self.multi_pass else "NO",
                     self.workers, cpu_info["threads"], cpu_info["cores"])
        logger.info("Idioma: %s | Confianza mín: %.0f%%", self.language, self.min_confidence)

        if self.multi_pass and preprocess:
            result = self._multi_pass(image_path, _preloaded)
        else:
            result = self._single_pass(image_path, preprocess)

        # Post-procesamiento de texto
        result = self._post_process(result)

        logger.info("RESULTADO FINAL: %d palabras, %d bloques, conf: %.1f%%",
                     result.word_count, len(result.blocks), result.avg_confidence)
        logger.info("═" * 50)
        return result

    # ─── Multi-paso optimizado (dos fases + early termination) ──
    def _multi_pass(
        self, image_path: str, _preloaded: Optional[Dict] = None,
    ) -> OCRResult:
        """
        Estrategia de dos fases con early termination.

        **Fase 1 (rápida):** ejecuta las 3 variantes prioritarias
        (V0=pipeline completo, V1=upscale puro, V2=CLAHE+sharpen)
        con los 2 mejores PSMs = ~6 llamadas a Tesseract.
        Si algún resultado supera ``_EARLY_STOP_SCORE``, se detiene.

        **Fase 2 (complementaria):** solo si Fase 1 no alcanzó un
        score suficiente, ejecuta las variantes restantes (binarizadas,
        examen) con los PSMs faltantes.

        Score = real_words × median_confidence

        Esta estrategia reduce de ~24 llamadas a ~4-6 en el caso
        típico (screenshots con texto legible), logrando ~5x speedup.
        """
        # ── Precarga de datos compartidos ──
        if _preloaded is None:
            _preloaded = {}
        if "img" not in _preloaded:
            _preloaded["img"] = smart_load(image_path)
        if "gray" not in _preloaded:
            _preloaded["gray"] = convert_to_grayscale(_preloaded["img"])
        if "is_ss" not in _preloaded:
            from image_processing import _detect_if_screenshot
            _preloaded["is_ss"] = _detect_if_screenshot(_preloaded["img"])
        if "up" not in _preloaded:
            _preloaded["up"] = upscale_for_ocr(_preloaded["gray"])

        variants = generate_preprocessing_variants(
            image_path, _preloaded=_preloaded,
        )

        # ── Determinar orden de PSMs ──
        if self.auto_psm:
            psm_order = analyze_optimal_psm(
                image_path, _preloaded=_preloaded,
            )
            logger.info("Auto-PSM: orden optimizado %s", psm_order)
        else:
            psm_order = self.CANDIDATE_PSMS

        # ── Separar variantes en Fase 1 (prioritarias) y Fase 2 ──
        # Las 3 primeras variantes son las más probables ganadoras
        PHASE1_VARIANTS = min(3, len(variants))
        phase1_variants = variants[:PHASE1_VARIANTS]
        phase2_variants = variants[PHASE1_VARIANTS:]

        # ── FASE 1A: variantes top × MEJOR PSM (batch mínimo) ──
        # Primero probar solo el PSM más probable para detectar early stop rápido
        best_psm = psm_order[0] if psm_order else 3
        phase1a_tasks: List[Tuple[int, np.ndarray, int]] = [
            (vi, variant, best_psm)
            for vi, variant in enumerate(phase1_variants)
        ]

        effective_workers = min(self.workers, len(phase1a_tasks))
        logger.info("Fase 1A: %d tareas (V0-V2 × PSM%d) → %d workers",
                     len(phase1a_tasks), best_psm, effective_workers)

        best, best_score, completed = self._run_phase(
            phase1a_tasks, effective_workers,
        )

        logger.info("Fase 1A completada: %d intentos, score=%.0f",
                     completed, best_score)

        # ── FASE 1B: mismo set × PSMs restantes (si Phase 1A insuficiente) ──
        remaining_psms = [p for p in psm_order if p != best_psm]
        if best_score < self._EARLY_STOP_SCORE and remaining_psms:
            phase1b_tasks: List[Tuple[int, np.ndarray, int]] = []
            for vi, variant in enumerate(phase1_variants):
                for psm in remaining_psms:
                    phase1b_tasks.append((vi, variant, psm))

            effective_workers_1b = min(self.workers, len(phase1b_tasks))
            logger.info("Fase 1B: %d tareas → %d workers",
                         len(phase1b_tasks), effective_workers_1b)

            best1b, score1b, completed1b = self._run_phase(
                phase1b_tasks, effective_workers_1b,
            )
            completed += completed1b
            if best1b is not None and score1b > best_score:
                best = best1b
                best_score = score1b
        else:
            logger.info("Score suficiente tras Fase 1A → omitiendo Fase 1B")

        logger.info("Fase 1 total: score=%.0f", best_score)

        # ── FASE 2 (solo si score insuficiente) ──
        if best_score < self._EARLY_STOP_SCORE and phase2_variants:
            logger.info("Score Fase 1 (%.0f) < umbral (%.0f) → ejecutando Fase 2...",
                         best_score, self._EARLY_STOP_SCORE)

            phase2_tasks: List[Tuple[int, np.ndarray, int]] = []
            for vi_offset, variant in enumerate(phase2_variants):
                vi = PHASE1_VARIANTS + vi_offset
                for psm in psm_order:
                    phase2_tasks.append((vi, variant, psm))

            effective_workers2 = min(self.workers, len(phase2_tasks))
            logger.info("Fase 2: %d tareas → %d workers", len(phase2_tasks), effective_workers2)

            best2, best_score2, completed2 = self._run_phase(phase2_tasks, effective_workers2)
            completed += completed2

            if best2 is not None and best_score2 > best_score:
                best = best2
                best_score = best_score2
                logger.info("Fase 2 mejoró score: %.0f", best_score)
        else:
            logger.info("Score Fase 1 suficiente (%.0f ≥ %.0f) → omitiendo Fase 2",
                         best_score, self._EARLY_STOP_SCORE)

        logger.info("Multi-paso total: %d intentos, score final: %.0f",
                     completed, best_score)

        if best is None:
            return OCRResult(language=self.language)
        return best

    def _run_phase(
        self,
        tasks: List[Tuple[int, np.ndarray, int]],
        workers: int,
    ) -> Tuple[Optional[OCRResult], float, int]:
        """
        Ejecuta un conjunto de tareas OCR en paralelo con early termination.

        Args:
            tasks: Lista de (variant_idx, variant_image, psm).
            workers: Número de workers paralelos.

        Returns:
            Tupla (mejor_resultado, mejor_score, tareas_completadas).
        """
        best: Optional[OCRResult] = None
        best_score = -1.0
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
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

                    # ── Early termination ──
                    if best_score >= self._EARLY_STOP_SCORE:
                        # Cancelar tareas pendientes
                        for f in futures:
                            f.cancel()
                        logger.info("Early stop: score %.0f ≥ %.0f",
                                     best_score, self._EARLY_STOP_SCORE)
                        break

                except Exception as exc:
                    logger.debug("V%d PSM%d falló: %s", vi, psm, exc)

        return best, best_score, completed

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
        """Filtra palabras por confianza y limpia basura.

        Incluye detección de radio buttons por geometría:
        palabras cuyo bounding box es aproximadamente cuadrado
        (width ≈ height) y cuyo texto es "O", "0", "o", "()"
        se marcan internamente como candidatas a radio button
        para el post-procesamiento posterior.
        """
        words: List[WordData] = []
        n = len(data["text"])

        for i in range(n):
            text = str(data["text"][i]).strip()
            conf = float(data["conf"][i])

            if not text or conf < self.min_confidence:
                continue

            # Filtrar solo basura inequívoca (mínimo posible)
            if len(text) == 1 and text in "|~`^\x00":
                continue

            # ── Detección de radio buttons por geometría ──
            # Los radio buttons en exámenes web son círculos
            # pequeños, así que su bounding box es casi cuadrado
            # y Tesseract los lee como "O", "0", "o", "()", "Oo"
            w_px = int(data["width"][i])
            h_px = int(data["height"][i])
            is_square = (0.7 <= w_px / max(h_px, 1) <= 1.4) if h_px > 0 else False
            is_small_circle_text = text in (
                "O", "o", "0", "Oo", "oO", "OO", "oo",
                "()", "()", "[]", "C", "Q", "G",
                "©", "®", "™", "¢",  # símbolos confundidos con círculos
            )

            if is_square and is_small_circle_text and w_px < 60:
                # Reemplazar directamente por marcador de radio button
                # que el post-procesamiento reconocerá
                text = "\u25cb"  # ○

            words.append(WordData(
                text=text, confidence=conf,
                x=int(data["left"][i]), y=int(data["top"][i]),
                width=w_px, height=h_px,
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

# ─── Símbolos de formulario web ───────────────
# Caracteres que Tesseract confunde cuando intenta leer
# los radio buttons (círculos vacíos) de formularios web.
# Estos se normalizan al símbolo estándar ○ (U+25CB)
_RADIO_BUTTON_SYMBOL = "○"  # WHITE CIRCLE U+25CB
_CHECKBOX_UNCHECKED = "☐"   # BALLOT BOX U+2610
_CHECKBOX_CHECKED = "☑"     # BALLOT BOX WITH CHECK U+2611

# Patrones que Tesseract genera al leer radio buttons
_RADIO_CONFUSIONS = {
    "©", "®", "™", "¢", "Ø", "ø",  # símbolos confundidos con círculos
}

# ─── Whitelist de palabras frecuentes que empiezan con O/o ───
# Estas palabras NO deben ser tratadas como radio button + texto.
# Solo se consideran los primeros 2-4 caracteres para matching rápido.
# Si la línea empieza con una de estas palabras, se deja intacta.
_O_WORD_PREFIXES: set = {
    # ── Palabras comunes en español que empiezan con O ──
    "Ob", "Oc", "Of", "Oh", "Oi", "Oj", "Ok", "Ol", "Om",
    "On", "Op", "Oq", "Os", "Ou", "Ov", "Ow", "Ox", "Oy", "Oz",
    "ob", "oc", "od", "of", "oh", "oi", "oj", "ok", "ol", "om",
    "on", "op", "oq", "os", "ou", "ov", "ow", "ox", "oy", "oz",
    # ── Bigramas de alta frecuencia que NO son radio button ──
    # "Or" y "or" son ambiguos (opción "r" vs palabra "Orden")
    # Se resuelven por contexto más abajo.
    # "Ot" se incluye porque "Otro/Otra" es muy frecuente
    "Ot", "ot",
}

# Palabras completas que empiezan con "Or"/"or" y NO son radio button.
# Si el texto tras "O" forma una de estas, se preserva la palabra.
_OR_SAFE_WORDS: set = {
    "Orden", "orden", "Ordena", "ordena", "Ordenar", "ordenar",
    "Ordenado", "ordenado", "Ordenamiento", "ordenamiento",
    "Origen", "origen", "Original", "original",
    "Orientación", "orientación", "Orientado", "orientado",
    "Oración", "oración",
    "Organizar", "organizar", "Organización", "organización",
    "Oro", "oro",
}


def _split_radio_prefix(line: str) -> str:
    """
    Separa un prefijo de radio button del texto de la opción.

    Detecta cuando Tesseract ha leído el círculo de un radio button
    como "O", "o" o "0" y lo ha concatenado (con o sin espacio) al
    texto real de la opción. Reemplaza el prefijo por ○ + espacio.

    Lógica de decisión (en orden):
      1. Línea vacía o de 1 carácter → no tocar.
      2. Ya empieza con ○/☐/☑ → no tocar.
      3. Empieza con símbolo confundido (©®™¢Øø) → reemplazar.
      4. Empieza con O/o/0 seguido de espacio y texto → reemplazar
         EXCEPTO si el texto que sigue forma una palabra española
         conocida con el prefijo (e.g., "Otro", "Obtener").
      5. Empieza con O/o/0 concatenado SIN espacio al texto → separar
         y reemplazar, EXCEPTO si forma palabra española conocida.

    Args:
        line: Línea de texto individual.

    Returns:
        Línea con el prefijo corregido, o la línea original si
        no se detectó radio button.
    """
    import re

    stripped = line.strip()
    if len(stripped) <= 1:
        return line

    # ── Ya tiene marcador de radio/checkbox → preservar ──
    if stripped[0] in ("○", "☐", "☑"):
        return line

    # ── Preservar indentación original ──
    indent = len(line) - len(line.lstrip())
    prefix_spaces = line[:indent]

    # ── Símbolo confundido con círculo al inicio → ○ ──
    for sym in _RADIO_CONFUSIONS:
        if stripped.startswith(sym):
            rest = stripped[len(sym):].lstrip()
            if rest:
                return f"{prefix_spaces}○ {rest}"
            return f"{prefix_spaces}○"

    # ── No empieza con O/o/0 → no es radio button ──
    if stripped[0] not in ("O", "o", "0"):
        return line

    # Extraer el primer carácter y el resto
    first_char = stripped[0]
    rest_raw = stripped[1:]  # todo lo que viene después de O/o/0

    # ── CASO 1: "O " + texto  (con espacio) ──
    # Ejemplo: "O .csv", "O Texto de opción", "O r"
    if rest_raw and rest_raw[0] == " ":
        rest_text = rest_raw.lstrip()
        if not rest_text:
            return line  # "O " sola → no tocar

        # Verificar si "O" + rest_text forma palabra conocida
        # (esto no aplica con espacio, la "O" es claramente separada)
        # Solo verificar "Or" + espacio ya que "Or" es ambiguo
        if first_char == "O" and rest_text and rest_text[0] == "r":
            # ¿El texto después forma una palabra "Or..." conocida?
            candidate = "O" + rest_text.split()[0] if rest_text else ""
            if candidate in _OR_SAFE_WORDS:
                return line  # "O rienta..." → "Orienta..." (no radio)

        return f"{prefix_spaces}○ {rest_text}"

    # ── CASO 2: "O" pegado al texto sin espacio ──
    # Ejemplos: "Oo", "Or", "0txt", "O.csv"
    if not rest_raw:
        return line  # Solo "O" aislada → no tocar (podría ser respuesta)

    # Sub-caso 2a: "O" + punto/coma/extensión → "○ .csv", "○ .txt"
    # Match: "O.xxx" donde xxx es extensión de archivo o similar
    m_ext = re.match(r'^[\.,:;](.*)$', rest_raw)
    if m_ext:
        return f"{prefix_spaces}○ {rest_raw}"

    # Sub-caso 2b: "0txt" → dígito cero + texto
    if first_char == "0" and rest_raw and rest_raw[0].isalpha():
        # "0txt" es claramente 0 + txt (no hay palabra "0txt")
        return f"{prefix_spaces}○ {rest_raw}"

    # Sub-caso 2c: "O" + letra → verificar si forma palabra conocida
    if rest_raw and rest_raw[0].isalpha():
        two_char = first_char + rest_raw[0]  # e.g., "Oo", "Or", "Ot"

        # Verificar en whitelist de prefijos seguros (NO radio button)
        if two_char in _O_WORD_PREFIXES:
            return line  # "Otro", "Obtener", "Once", etc.

        # Caso especial: "Or..." — verificar palabra completa
        if two_char in ("Or", "or"):
            # Extraer la primera palabra completa
            word_match = re.match(r'^([a-záéíóúñA-ZÁÉÍÓÚÑ]+)', rest_raw)
            if word_match:
                candidate = first_char + word_match.group(1)
                if candidate in _OR_SAFE_WORDS:
                    return line

        # Si el bigrama NO está en whitelist → es radio button
        # "Oo" → "○ o",  "Or" (no en safe words) → "○ r"
        return f"{prefix_spaces}○ {rest_raw}"

    # Sub-caso 2d: "O" + dígito → probablemente dato, no radio
    if rest_raw and rest_raw[0].isdigit():
        return line  # "O5" podría ser dato legítimo

    # ── Fallback: no modificar ──
    return line


def _fix_ocr_text(text: str) -> str:
    """
    Corrige errores frecuentes de OCR:
      - Comillas tipográficas → rectas
      - Ligaduras → caracteres separados
      - Espacios no rompibles → espacios normales
      - Caracteres confundidos en contexto de código
      - Espaciado alrededor de guiones bajos
      - Normalización de radio buttons y checkboxes
      - Limpieza de artefactos de compresión de imagen
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

    # 4. ─── Normalización de radio buttons ───
    # Usa _split_radio_prefix() por línea para detectar cuando
    # Tesseract leyó el círculo del radio button como "O", "o", "0"
    # y lo concatenó al texto de la opción (con o sin espacio).
    #
    # Ejemplos que se corrigen:
    #   "Oo"    → "○ o"       (O + o concatenados)
    #   "Or"    → "○ r"       (O + r concatenados, no es "Orden")
    #   "O .csv"→ "○ .csv"    (O + espacio + extensión)
    #   "0txt"  → "○ txt"     (cero + texto concatenado)
    #
    # Palabras legítimas se preservan:
    #   "Otro"  → "Otro"      (está en _O_WORD_PREFIXES: "Ot")
    #   "Orden" → "Orden"     (está en _OR_SAFE_WORDS)
    lines = text.split('\n')
    lines = [_split_radio_prefix(line) for line in lines]
    text = '\n'.join(lines)

    # "()" o "[]" vacíos al inicio de línea → ○ / ☐
    text = re.sub(
        r'^\(\)\s*',
        f'{_RADIO_BUTTON_SYMBOL} ',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^\[\]\s*',
        f'{_CHECKBOX_UNCHECKED} ',
        text,
        flags=re.MULTILINE,
    )

    # 5. ─── Normalización de checkboxes ───
    # "[x]" o "[X]" o "[v]" → ☑
    text = re.sub(
        r'\[[xXvV✓✔]\]\s*',
        f'{_CHECKBOX_CHECKED} ',
        text,
    )

    # 6. ─── Limpieza de artefactos de compresión ───
    # Caracteres sueltos que son ruido de compresión JPEG
    # (aparecen como puntos, guiones, tildes aislados entre palabras)
    text = re.sub(r'\s[¢©®™]\s', ' ', text)

    # 7. Colapsar espacios múltiples internos (>3) a uno solo
    #    pero preservar indentación al inicio de línea
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        # Preservar espacios de indentación al inicio
        stripped = line.lstrip(' ')
        indent = len(line) - len(stripped)
        # En el contenido, colapsar 3+ espacios a 1
        stripped = re.sub(r' {3,}', ' ', stripped)
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
      - Detecta múltiples formatos de enumeración de preguntas.
      - Normaliza radio buttons y checkboxes.

    Args:
        text: Texto completo del documento reconstruido.

    Returns:
        Texto mejorado con formato de examen.
    """
    import re

    lines = text.split('\n')

    # ── Detectar si es un examen/cuestionario ──
    # Patrones de detección de preguntas ampliados:
    #   - "Pregunta N" / "Pregunta N:" / "Pregunta N."
    #   - "N." / "N)" / "N.-" al inicio de línea (numeración)
    #   - "Q1:" / "Q1." (formato inglés)
    #   - Líneas con ○ o radio buttons (opciones de respuesta)
    question_indices = []
    radio_count = 0

    _QUESTION_HEAD_RE = re.compile(
        r'^\s*('
        r'Pregunta\s+\d+'
        r'|\d{1,3}[\.\)\-]\s+[A-ZÁÉÍÓÚÑ¿¡]'
        r'|Q\d{1,3}[\.:\)]\s'
        r')',
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        if _QUESTION_HEAD_RE.match(stripped):
            question_indices.append(i)
        # Contar líneas que parecen opciones de respuesta
        if re.match(r'^\s*[○☐☑]\s', stripped):
            radio_count += 1
        elif re.match(r'^\s*[a-dA-D][\)\.]\s', stripped):
            radio_count += 1

    # Considerar examen si hay ≥2 preguntas detectadas,
    # O si hay muchas opciones tipo radio/checkbox (≥4)
    is_exam = len(question_indices) >= 2 or radio_count >= 4

    if not is_exam:
        # No es examen — solo correcciones globales
        text = _apply_global_fixes(text)
        return text

    # Si no se detectaron headers pero sí opciones radio,
    # hacer correcciones globales sin reestructurar el layout
    if not question_indices:
        text = _normalize_option_lines(text)
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


def _normalize_option_lines(text: str) -> str:
    """
    Normaliza líneas de opciones de respuesta cuando no se
    detectaron preguntas explícitas (formato libre o parcial).

    Busca líneas que parecen opciones y les agrega ○ si falta.
    Delega la detección de radio buttons concatenados a
    ``_split_radio_prefix()`` para evitar falsos positivos.

    Args:
        text: Texto completo del documento.

    Returns:
        Texto con opciones normalizadas.
    """
    import re
    lines = text.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()

        # Ya tiene radio button → preservar
        if re.match(r'^[○☐☑]\s', stripped):
            result.append(line)
            continue

        # Patrón a)/b)/c)/d) → agregar ○
        m = re.match(r'^[a-dA-D][\)\.]\s+(.+)', stripped)
        if m:
            result.append(f'○ {m.group(1)}')
            continue

        # Delegar detección de O/o/0 + texto a _split_radio_prefix()
        # que ya maneja concatenaciones y whitelist de palabras
        fixed = _split_radio_prefix(line)
        result.append(fixed)

    return '\n'.join(result)


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

    Detecta patrones de Python, JavaScript, Java, C/C++ y otros
    lenguajes comunes en exámenes de programación.

    Args:
        stripped: Línea sin espacios al inicio/final.

    Returns:
        True si parece código fuente.
    """
    import re

    # Patrones inequívocos de código
    code_patterns = [
        # Python keywords
        r'^(if|elif|else|for|while|def|class|return|import|from|try|except|finally|with)\b',
        r'^[a-z_]\w*\s*=\s*',                           # asignación: var = ...
        r'^\w+\(.*\)',                                    # llamada: func(...)
        r'print\s*\(',                                    # print(
        r'input\s*\(',                                    # input(
        r'=\s*input\s*\(',                                # = input(
        r'^\s*#\s*\w',                                    # comentario Python: # algo
        r'\brange\s*\(',                                  # range(
        r'\blen\s*\(',                                    # len(
        r'\bint\s*\(',                                    # int(
        r'\bstr\s*\(',                                    # str(
        r'\bfloat\s*\(',                                  # float(
        r'\blist\s*\(',                                   # list(
        r'\bdict\s*\(',                                   # dict(
        r'\b(True|False|None)\b',                         # constantes Python
        r'^\s*\w+\s*\[.*\]\s*=',                         # indexación: arr[i] = ...
        # Patrones genéricos de programación
        r'\{\s*$',                                        # apertura de bloque {
        r'^\s*\}',                                        # cierre de bloque }
        r';\s*$',                                         # terminador de sentencia ;
        r'^\s*(var|let|const|int|float|double|string|char|void|public|private|static)\s',
        r'System\.out',                                   # Java
        r'Console\.(Write|Read)',                          # C#
        r'cout\s*<<',                                     # C++
        r'cin\s*>>',                                      # C++
        r'printf\s*\(',                                   # C
        r'scanf\s*\(',                                    # C
    ]

    for pat in code_patterns:
        if re.search(pat, stripped):
            return True

    # Contiene operadores de programación
    code_operators = ['==', '>=', '<=', '!=', '= input(', 'print(',
                      '+=', '-=', '*=', '/=', '%=', '=>', '->']
    return any(op in stripped for op in code_operators)



def _clean_option_text(text: str) -> str:
    """
    Limpia prefijos residuales del OCR en opciones de respuesta.

    Elimina marcadores de radio button, letras de opción, y
    artefactos que Tesseract introduce al leer formularios web.
    Maneja tanto prefijos separados por espacio como concatenados
    (ej: "Or" → "r", "0txt" → "txt", "O .csv" → ".csv").

    Se ejecuta DESPUÉS de que _split_radio_prefix() ya haya
    normalizado la línea, así que aquí solo se limpian residuos
    dentro del contexto de una opción ya clasificada.

    Args:
        text: Texto de una opción individual.

    Returns:
        Texto limpio de la opción.
    """
    import re

    # Quitar ○/☐/☑ ya existente si se duplicaría
    text = re.sub(r'^[○☐☑]\s*', '', text)

    # ── Separar O/o/0 concatenados con texto (casos residuales) ──
    # "Oo" → "o", "Or" → "r", "0txt" → "txt", "O.csv" → ".csv"
    # Solo si el bigrama NO es prefijo de palabra española conocida.
    m = re.match(r'^([Oo0])([a-záéíóúñ\\.].*)$', text)
    if m:
        prefix_char = m.group(1)
        rest = m.group(2)
        two_char = prefix_char + rest[0] if rest else ""

        # Si el bigrama está en whitelist de palabras españolas → no tocar
        if two_char not in _O_WORD_PREFIXES:
            # Verificar palabras "Or..." en safelist
            if two_char in ("Or", "or"):
                word_m = re.match(r'^([a-záéíóúñ]+)', rest, re.IGNORECASE)
                candidate = prefix_char + (word_m.group(1) if word_m else "")
                if candidate not in _OR_SAFE_WORDS:
                    text = rest
            else:
                text = rest

    # Quitar "O "/"o "/"0 " con espacio al inicio
    text = re.sub(r'^[Oo0]\s+', '', text)

    # Quitar letras como opción: a) b) c) d) A) B) C) D)
    text = re.sub(r'^[a-dA-D][\)\.]\s*', '', text)
    # Quitar O) o 0) que son confusiones de radio + paréntesis
    text = re.sub(r'^[Oo0][\)\.]\s*', '', text)

    # Quitar símbolos confundidos con radio buttons al inicio
    text = re.sub(r'^[©®™¢Øø]\s*', '', text)

    # Quitar "()" o "[]" residuales al inicio
    text = re.sub(r'^\(\)\s*', '', text)
    text = re.sub(r'^\[\]\s*', '', text)

    # Quitar guion aislado al inicio (a veces OCR lee "— opción")
    text = re.sub(r'^[\-–—]\s+', '', text)

    return text.strip()


def _apply_global_fixes(text: str) -> str:
    """
    Correcciones globales de OCR que aplican a todo el documento.

    Incluye:
      - Confusiones I/l comunes en OCR (Ist→lst)
      - Fusión de palabras comunes en español
      - Confusiones de letras/números frecuentes
      - Limpieza de artefactos repetitivos
      - Correcciones de acentos mal reconocidos

    Args:
        text: Texto completo.

    Returns:
        Texto corregido.
    """
    import re

    # ── Confusión I/l (muy común en OCR) ──
    text = re.sub(r'\bIst\b', 'lst', text)
    text = re.sub(r'\bIst\(', 'lst(', text)
    text = re.sub(r'\bIst\[', 'lst[', text)
    text = re.sub(r'\bIen\b', 'len', text)
    text = re.sub(r'\bIen\(', 'len(', text)

    # ── Fusión de palabras y confusiones l/(/{ ──
    text = re.sub(r'\biflint\b', 'if(int', text)
    text = re.sub(r'\bifllint\b', 'if(int', text)
    text = re.sub(r'\bprinl\b', 'print', text)
    text = re.sub(r'\bprintt\b', 'print', text)
    text = re.sub(r'\binpul\b', 'input', text)

    # ── M/P → M/F (contexto género) ──
    text = re.sub(r'Genero\s*\(M/P\)', 'Genero (M/F)', text)
    text = re.sub(r'\(M/P\)', '(M/F)', text)

    # ── Palabras pegadas frecuentes en español ──
    text = re.sub(r'\besun\b', 'es un', text)
    text = re.sub(r'\besuna\b', 'es una', text)
    text = re.sub(r'\bnoexiste\b', 'no existe', text)
    text = re.sub(r'\bnoes\b', 'no es', text)
    text = re.sub(r'\bdela\b(?!nte)', 'de la', text)  # dela pero no delante
    text = re.sub(r'\bdelos\b', 'de los', text)
    text = re.sub(r'\bporque(?:es)\b', 'porque es', text)
    text = re.sub(r'\bque es\b', 'que es', text)  # ya correcto, preservar

    # ── Acentos mal reconocidos ──
    text = re.sub(r'\bdiferencía\b', 'diferencia', text)
    text = re.sub(r'\bfuncíón\b', 'función', text)
    text = re.sub(r'\boperacíón\b', 'operación', text)
    text = re.sub(r'\bcondícíón\b', 'condición', text)
    text = re.sub(r'\bvaríable\b', 'variable', text)

    # ── Confusiones número/letra comunes ──
    # "l" (L minúscula) como "1" en contexto de código
    text = re.sub(r'(?<=\d)l(?=\d)', '1', text)  # 1l0 → 110
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)  # 1O0 → 100

    # ── Artefactos repetitivos de OCR ──
    # Secuencias de puntos/guiones excesivos (>10) → limpiar
    text = re.sub(r'\.{10,}', '...', text)
    text = re.sub(r'\-{10,}', '─' * 10, text)
    text = re.sub(r'_{10,}', '_' * 10, text)

    # ── Limpieza de líneas que son puro ruido ──
    # Líneas con solo 1-2 caracteres de ruido (no letras/dígitos)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Preservar líneas vacías y líneas con contenido real
        if not stripped or len(stripped) > 2:
            cleaned.append(line)
            continue
        # Líneas de 1-2 chars: mantener solo si son alfanuméricas
        if re.match(r'^[a-zA-Z0-9áéíóúñÁÉÍÓÚÑ]+$', stripped):
            cleaned.append(line)
        # Else: descartar líneas de 1-2 chars con solo símbolos (ruido)
    text = '\n'.join(cleaned)

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
