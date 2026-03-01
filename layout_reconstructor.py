"""
layout_reconstructor.py — Reconstrucción de estructura visual del documento.

Analiza la distribución espacial de los bloques de texto detectados
para reconstruir columnas, párrafos, encabezados y la jerarquía
visual original del documento.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ocr_engine import BlockData, LineData, OCRResult, _post_process_document
from utils import logger


# ─────────────────────────────────────────────
# Estructuras de layout
# ─────────────────────────────────────────────
@dataclass
class Column:
    """Representa una columna visual detectada en el documento."""
    column_id: int
    x_start: int
    x_end: int
    blocks: List[BlockData] = field(default_factory=list)


@dataclass
class ReconstructedDocument:
    """Documento reconstruido con estructura visual preservada."""
    columns: List[Column] = field(default_factory=list)
    ordered_blocks: List[BlockData] = field(default_factory=list)
    formatted_text: str = ""
    is_multicolumn: bool = False
    column_count: int = 1


# ─────────────────────────────────────────────
# Reconstructor principal
# ─────────────────────────────────────────────
class LayoutReconstructor:
    """
    Reconstruye la estructura visual del documento a partir de
    los datos de OCR.

    Detecta columnas, párrafos y organiza el texto en el orden
    de lectura correcto.

    Atributos:
        column_gap_threshold: Distancia horizontal mínima (en px)
            para considerar que hay dos columnas separadas.
        line_gap_multiplier: Multiplicador del interlineado promedio
            para detectar separación entre párrafos.
    """

    def __init__(
        self,
        column_gap_threshold: float = 0.15,
        line_gap_multiplier: float = 1.8,
    ) -> None:
        """
        Args:
            column_gap_threshold: Fracción del ancho de imagen como umbral
                para separación de columnas (0.0 - 1.0).
            line_gap_multiplier: Factor para detectar saltos de párrafo.
        """
        self.column_gap_threshold = column_gap_threshold
        self.line_gap_multiplier = line_gap_multiplier

    def reconstruct(self, ocr_result: OCRResult) -> ReconstructedDocument:
        """
        Reconstruye la estructura completa del documento.

        Args:
            ocr_result: Resultado de la extracción OCR.

        Returns:
            ReconstructedDocument con la estructura visual preservada.
        """
        logger.info("Reconstruyendo estructura del documento...")

        if not ocr_result.blocks:
            logger.warning("No se detectaron bloques de texto.")
            return ReconstructedDocument(formatted_text="[Sin texto detectado]")

        # Paso 1: Detectar columnas
        columns = self._detect_columns(
            ocr_result.blocks, ocr_result.image_width
        )

        is_multi = len(columns) > 1
        if is_multi:
            logger.info("Documento multi-columna detectado: %d columnas", len(columns))
        else:
            logger.info("Documento de una sola columna.")

        # Paso 2: Ordenar bloques dentro de cada columna
        ordered_blocks = self._order_blocks(columns)

        # Paso 3: Reconstruir texto con formato
        formatted_text = self._format_text(ordered_blocks, ocr_result.image_width)

        # Paso 4: Post-procesamiento a nivel de documento
        #         (detecta exámenes, agrega ○ a opciones, separadores, etc.)
        formatted_text = _post_process_document(formatted_text)

        doc = ReconstructedDocument(
            columns=columns,
            ordered_blocks=ordered_blocks,
            formatted_text=formatted_text,
            is_multicolumn=is_multi,
            column_count=len(columns),
        )

        logger.info("Estructura reconstruida: %d bloques ordenados.", len(ordered_blocks))
        return doc

    def _detect_columns(
        self,
        blocks: List[BlockData],
        image_width: int,
    ) -> List[Column]:
        """
        Detecta columnas analizando la distribución horizontal de los bloques.

        Algoritmo:
          1. Calcular el centro X de cada bloque.
          2. Ordenar por posición X.
          3. Agrupar bloques cuyo centro X esté dentro del umbral.

        Args:
            blocks: Bloques detectados por OCR.
            image_width: Ancho de la imagen en píxeles.

        Returns:
            Lista de Column ordenadas de izquierda a derecha.
        """
        if not blocks:
            return []

        gap_px = int(image_width * self.column_gap_threshold)

        # Calcular centro X de cada bloque
        block_centers = []
        for block in blocks:
            cx = block.x + block.width // 2
            block_centers.append((cx, block))

        # Ordenar por centro X
        block_centers.sort(key=lambda item: item[0])

        # Agrupar en columnas
        columns: List[Column] = []
        current_column_blocks: List[BlockData] = [block_centers[0][1]]
        current_x_min = block_centers[0][1].x
        current_x_max = block_centers[0][1].x + block_centers[0][1].width

        for i in range(1, len(block_centers)):
            cx, block = block_centers[i]
            prev_cx = block_centers[i - 1][0]

            # Si la distancia entre centros X supera el umbral → nueva columna
            if cx - prev_cx > gap_px:
                col = Column(
                    column_id=len(columns),
                    x_start=current_x_min,
                    x_end=current_x_max,
                    blocks=current_column_blocks,
                )
                columns.append(col)
                current_column_blocks = [block]
                current_x_min = block.x
                current_x_max = block.x + block.width
            else:
                current_column_blocks.append(block)
                current_x_min = min(current_x_min, block.x)
                current_x_max = max(current_x_max, block.x + block.width)

        # Última columna
        col = Column(
            column_id=len(columns),
            x_start=current_x_min,
            x_end=current_x_max,
            blocks=current_column_blocks,
        )
        columns.append(col)

        return columns

    def _order_blocks(self, columns: List[Column]) -> List[BlockData]:
        """
        Ordena los bloques siguiendo el orden natural de lectura:
        columnas de izquierda a derecha, y dentro de cada columna
        de arriba hacia abajo.

        Args:
            columns: Columnas detectadas.

        Returns:
            Lista de bloques en orden de lectura.
        """
        ordered: List[BlockData] = []

        for column in columns:
            sorted_blocks = sorted(column.blocks, key=lambda b: b.y)
            ordered.extend(sorted_blocks)

        return ordered

    def _format_text(
        self,
        blocks: List[BlockData],
        image_width: int,
    ) -> str:
        """
        Genera texto formateado a partir de los bloques ordenados.

        Preserva:
          - Saltos de línea dentro de bloques.
          - Separación entre bloques (doble salto).
          - Indentación relativa basada en posición X.

        Args:
            blocks: Bloques ordenados en orden de lectura.
            image_width: Ancho de la imagen para cálculos relativos.

        Returns:
            Texto formateado listo para copiar y pegar.
        """
        parts: List[str] = []

        for block in blocks:
            block_lines = self._format_block_lines(block, image_width)
            parts.append(block_lines)

        return "\n\n".join(parts)

    def _format_block_lines(
        self,
        block: BlockData,
        image_width: int,
    ) -> str:
        """
        Formatea las líneas de un bloque individual.

        Detecta si una línea debe tener indentación basándose
        en su posición X relativa al bloque.

        Args:
            block: Bloque de texto.
            image_width: Ancho de la imagen.

        Returns:
            Texto del bloque con formato preservado.
        """
        if not block.lines:
            return ""

        # Calcular X mínimo del bloque como referencia
        min_x = min(line.x for line in block.lines)

        formatted_lines: List[str] = []
        prev_y_end: Optional[int] = None

        for line in block.lines:
            # Detectar saltos de párrafo dentro del bloque
            if prev_y_end is not None:
                gap = line.y - prev_y_end
                avg_line_height = line.height if line.height > 0 else 20
                if gap > avg_line_height * self.line_gap_multiplier:
                    formatted_lines.append("")  # Línea vacía = párrafo

            # Calcular indentación relativa
            indent_px = line.x - min_x
            indent_chars = self._px_to_indent(indent_px, image_width)
            indent = " " * indent_chars

            formatted_lines.append(f"{indent}{line.text}")
            prev_y_end = line.y + line.height

        return "\n".join(formatted_lines)

    @staticmethod
    def _px_to_indent(px: int, image_width: int, max_indent: int = 20) -> int:
        """
        Convierte una distancia en píxeles a espacios de indentación.

        Args:
            px: Distancia en píxeles desde el borde izquierdo del bloque.
            image_width: Ancho total de la imagen.
            max_indent: Máximo de espacios de indentación.

        Returns:
            Número de espacios de indentación.
        """
        if px < 15 or image_width == 0:
            return 0

        # Proporción relativa al ancho de imagen
        ratio = px / image_width
        indent = int(ratio * 80)  # 80 columnas como referencia
        return min(indent, max_indent)


# ─────────────────────────────────────────────
# Análisis de estructura
# ─────────────────────────────────────────────
def analyze_document_structure(ocr_result: OCRResult) -> Dict:
    """
    Genera un análisis resumido de la estructura del documento.

    Args:
        ocr_result: Resultado OCR.

    Returns:
        Diccionario con métricas del documento.
    """
    total_lines = sum(len(b.lines) for b in ocr_result.blocks)
    total_words = ocr_result.word_count

    # Detectar posibles encabezados (líneas con pocas palabras y alto Y)
    headers = []
    for block in ocr_result.blocks:
        for line in block.lines:
            words_in_line = len(line.words)
            if words_in_line <= 6 and line.y < ocr_result.image_height * 0.15:
                headers.append(line.text)

    return {
        "total_bloques": len(ocr_result.blocks),
        "total_lineas": total_lines,
        "total_palabras": total_words,
        "confianza_promedio": round(ocr_result.avg_confidence, 1),
        "idioma": ocr_result.language,
        "dimensiones": f"{ocr_result.image_width}x{ocr_result.image_height}",
        "posibles_encabezados": headers,
    }
