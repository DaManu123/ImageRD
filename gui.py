"""
gui.py — Interfaz gráfica profesional para ImageRD.

Proporciona una interfaz visual moderna con CustomTkinter que expone
todas las opciones del pipeline OCR de forma intuitiva.

Características:
  - Panel de control con todas las opciones del CLI.
  - Vista previa de la imagen seleccionada.
  - Área de texto con el resultado OCR.
  - Botón de copiar texto rápido.
  - Estadísticas en tiempo real.
  - Barra de progreso con estado del procesamiento.
  - Modo oscuro / claro.

Uso:
    python gui.py
"""

import os
import io
import sys
import time
import logging
import platform
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import List, Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

from utils import (
    logger,
    validate_image_path,
    validate_output_format,
    SUPPORTED_FORMATS,
)
from ocr_engine import OCREngine, OCRResult, get_cpu_info, get_optimal_workers
from layout_reconstructor import LayoutReconstructor, ReconstructedDocument
from exporters import export_all


# ─────────────────────────────────────────────
# Constantes de la GUI
# ─────────────────────────────────────────────
APP_TITLE = "ImageRD"
APP_SUBTITLE = "Sistema OCR Profesional"
APP_VERSION = "1.0.0"

WINDOW_WIDTH = 1120
WINDOW_HEIGHT = 740
MIN_WIDTH = 920
MIN_HEIGHT = 620
SIDEBAR_WIDTH = 330

# Idiomas disponibles para OCR
LANGUAGES = {
    "Español": "spa",
    "English": "eng",
    "Español + English": "spa+eng",
    "Português": "por",
    "Français": "fra",
    "Deutsch": "deu",
}

# Modos PSM con descripciones legibles
PSM_MODES = {
    "3 — Automático (recomendado)": 3,
    "6 — Bloque de texto uniforme": 6,
    "4 — Columna de texto": 4,
    "1 — Automático con OSD": 1,
    "7 — Línea de texto": 7,
    "8 — Palabra individual": 8,
    "11 — Texto disperso": 11,
    "13 — Línea cruda": 13,
}

# Tipos de archivo para el diálogo de selección
IMAGE_FILETYPES = [
    ("Imágenes", "*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("WebP", "*.webp"),
    ("TIFF", "*.tiff *.tif"),
    ("BMP", "*.bmp"),
    ("Todos", "*.*"),
]

# Colores personalizados
COLOR_ACCENT = ("#2563eb", "#3b82f6")
COLOR_ACCENT_HOVER = ("#1d4ed8", "#2563eb")
COLOR_SUCCESS = ("#16a34a", "#22c55e")
COLOR_SURFACE = ("gray92", "gray17")


# ─────────────────────────────────────────────
# Handler de logging para la barra de estado
# ─────────────────────────────────────────────
class _GUILogHandler(logging.Handler):
    """
    Captura mensajes del logger del proyecto y los reenvía
    a la barra de estado de la GUI vía callback.
    """

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        msg = self.format(record)
        try:
            self.callback(msg)
        except Exception:
            pass


# ─────────────────────────────────────────────
# Aplicación principal
# ─────────────────────────────────────────────
class ImageRDApp(ctk.CTk):
    """
    Ventana principal de ImageRD.

    Layout:
      ┌─────────────┬──────────────────────────────┐
      │  Sidebar     │  Área de resultados           │
      │  (controles) │  (texto + stats + acciones)   │
      ├─────────────┴──────────────────────────────┤
      │  Barra de estado (progreso + mensajes)      │
      └─────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()

        # ── Configuración de ventana ──
        self.title(f"{APP_TITLE} — {APP_SUBTITLE}")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(MIN_WIDTH, MIN_HEIGHT)

        # Centrar en pantalla
        self.update_idletasks()
        x = (self.winfo_screenwidth() - WINDOW_WIDTH) // 2
        y = (self.winfo_screenheight() - WINDOW_HEIGHT) // 2
        self.geometry(f"+{x}+{y}")

        # ── Estado interno ──
        self._image_path: Optional[str] = None
        self._ocr_result: Optional[OCRResult] = None
        self._reconstructed: Optional[ReconstructedDocument] = None
        self._generated_files: List[Path] = []
        self._processing = False
        self._temp_files: List[Path] = []  # archivos temporales del portapapeles

        # ── Tema por defecto ──
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Construir interfaz ──
        self._setup_grid()
        self._build_sidebar()
        self._build_main_area()
        self._build_status_bar()

        # ── Conectar log handler a la barra de estado ──
        self._log_handler = _GUILogHandler(self._on_log_message)
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(self._log_handler)

        # ── Atajo de teclado: Ctrl+V para pegar imagen ──
        self.bind("<Control-v>", lambda e: self._paste_image_from_clipboard())

        # ── Manejar cierre de ventana ──
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Verificar Tesseract al iniciar ──
        self.after(500, self._check_tesseract)

    # ─────────────────────────────────────────
    # Grid principal
    # ─────────────────────────────────────────
    def _setup_grid(self):
        """Configura las proporciones del grid de la ventana."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0)  # sidebar fija
        self.grid_columnconfigure(1, weight=1)  # main se expande

    # ─────────────────────────────────────────
    # SIDEBAR — Panel de controles
    # ─────────────────────────────────────────
    def _build_sidebar(self):
        """Construye el panel lateral con todos los controles."""
        self.sidebar = ctk.CTkScrollableFrame(
            self, width=SIDEBAR_WIDTH, corner_radius=0,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        r = 0  # contador de filas

        # ── Encabezado ──
        ctk.CTkLabel(
            self.sidebar, text=f"🖼  {APP_TITLE}",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=r, column=0, padx=20, pady=(20, 0)); r += 1

        ctk.CTkLabel(
            self.sidebar,
            text=f"v{APP_VERSION} — {APP_SUBTITLE}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).grid(row=r, column=0, padx=20, pady=(0, 16)); r += 1

        # ── IMAGEN DE ENTRADA ──
        r = self._section_header(r, "IMAGEN DE ENTRADA")

        # Marco para thumbnail
        self._thumb_frame = ctk.CTkFrame(
            self.sidebar, height=130, corner_radius=8,
            fg_color=COLOR_SURFACE,
        )
        self._thumb_frame.grid(row=r, column=0, padx=20, pady=(4, 4), sticky="ew")
        self._thumb_frame.grid_propagate(False)
        self._thumb_frame.grid_rowconfigure(0, weight=1)
        self._thumb_frame.grid_columnconfigure(0, weight=1)

        self._thumb_label = ctk.CTkLabel(
            self._thumb_frame,
            text="Sin imagen seleccionada\n\n«Seleccionar» o Ctrl+V para pegar",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self._thumb_label.grid(row=0, column=0, padx=10, pady=10); r += 1

        # Información de la imagen
        self._img_info = ctk.CTkLabel(
            self.sidebar, text="",
            font=ctk.CTkFont(size=11), text_color="gray",
        )
        self._img_info.grid(row=r, column=0, padx=20, pady=(0, 4)); r += 1

        # Botón seleccionar
        self._browse_btn = ctk.CTkButton(
            self.sidebar, text="📂  Seleccionar imagen",
            command=self._browse_image, height=34,
        )
        self._browse_btn.grid(row=r, column=0, padx=20, pady=(0, 4), sticky="ew"); r += 1

        # Botón pegar desde portapapeles
        self._paste_btn = ctk.CTkButton(
            self.sidebar, text="📋  Pegar imagen  (Ctrl+V)",
            command=self._paste_image_from_clipboard, height=34,
            fg_color="transparent", border_width=1,
            text_color=("gray10", "gray90"),
            border_color=("gray50", "gray50"),
        )
        self._paste_btn.grid(row=r, column=0, padx=20, pady=(0, 16), sticky="ew"); r += 1

        # ── FORMATO DE SALIDA ──
        r = self._section_header(r, "FORMATO DE SALIDA")

        self._format_var = ctk.StringVar(value="txt")
        self._format_seg = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["txt", "json", "docx", "all"],
            variable=self._format_var,
            font=ctk.CTkFont(size=13),
        )
        self._format_seg.grid(row=r, column=0, padx=20, pady=(4, 16), sticky="ew"); r += 1

        # ── IDIOMA OCR ──
        r = self._section_header(r, "IDIOMA OCR")

        self._lang_var = ctk.StringVar(value="Español")
        ctk.CTkComboBox(
            self.sidebar,
            values=list(LANGUAGES.keys()),
            variable=self._lang_var,
            height=32, font=ctk.CTkFont(size=13),
        ).grid(row=r, column=0, padx=20, pady=(4, 16), sticky="ew"); r += 1

        # ── OPCIONES ──
        r = self._section_header(r, "OPCIONES")

        # Switch: Preprocesamiento
        self._preproc_var = ctk.BooleanVar(value=True)
        r = self._switch_row(r, "Preprocesamiento", self._preproc_var)

        # Switch: Multi-paso
        self._multipass_var = ctk.BooleanVar(value=True)
        r = self._switch_row(r, "Multi-paso (más preciso)", self._multipass_var)

        # Info: PSM y confianza auto-detectados
        ctk.CTkLabel(
            self.sidebar, text="PSM: auto-detección inteligente ✓",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).grid(row=r, column=0, padx=20, pady=(8, 2), sticky="w"); r += 1

        ctk.CTkLabel(
            self.sidebar, text="Confianza: adaptativa automática ✓",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).grid(row=r, column=0, padx=20, pady=(2, 12), sticky="w"); r += 1

        # ── PARALELISMO ──
        r = self._section_header(r, "PARALELISMO")

        _cpu_info = get_cpu_info()
        _cores = _cpu_info["cores"]
        _threads = _cpu_info["threads"]
        _optimal = get_optimal_workers()

        ctk.CTkLabel(
            self.sidebar,
            text=f"CPU: {_cores} cores / {_threads} hilos → {_optimal} workers",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).grid(row=r, column=0, padx=20, pady=(4, 2), sticky="w"); r += 1

        self._workers_var = ctk.IntVar(value=0)
        _worker_values = ["0 — Auto"] + [str(i) for i in range(1, _threads + 1)]
        self._workers_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=_worker_values,
            command=self._on_workers_change,
            height=30, font=ctk.CTkFont(size=11),
        )
        self._workers_menu.set("0 — Auto")
        self._workers_menu.grid(row=r, column=0, padx=20, pady=(0, 20), sticky="ew"); r += 1

        # ── BOTÓN PROCESAR ──
        self._process_btn = ctk.CTkButton(
            self.sidebar,
            text="🔍  PROCESAR IMAGEN",
            command=self._start_processing,
            height=48,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
        )
        self._process_btn.grid(row=r, column=0, padx=20, pady=(0, 8), sticky="ew"); r += 1

        # ── Toggle de tema ──
        self._theme_btn = ctk.CTkButton(
            self.sidebar,
            text="☀  Cambiar a modo claro",
            command=self._toggle_theme,
            height=28, font=ctk.CTkFont(size=11),
            fg_color="transparent", text_color="gray",
            hover_color=("gray80", "gray30"),
        )
        self._theme_btn.grid(row=r, column=0, padx=20, pady=(0, 20), sticky="ew"); r += 1

    def _section_header(self, row: int, text: str) -> int:
        """Agrega un encabezado de sección al sidebar. Devuelve la fila siguiente."""
        ctk.CTkLabel(
            self.sidebar, text=text,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray40", "gray55"),
        ).grid(row=row, column=0, padx=20, pady=(8, 0), sticky="w")
        return row + 1

    def _switch_row(self, row: int, label: str, variable: ctk.BooleanVar) -> int:
        """Agrega una fila con label + switch. Devuelve la fila siguiente."""
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.grid(row=row, column=0, padx=20, pady=(4, 2), sticky="ew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text=label, font=ctk.CTkFont(size=13),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkSwitch(
            frame, text="", variable=variable, width=46,
        ).grid(row=0, column=1, sticky="e")

        return row + 1

    # ─────────────────────────────────────────
    # MAIN — Área de resultados
    # ─────────────────────────────────────────
    def _build_main_area(self):
        """Construye el área principal con preview de texto y acciones."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.main_frame.grid_rowconfigure(1, weight=1)  # textbox se expande
        self.main_frame.grid_columnconfigure(0, weight=1)

        # ── Header ──
        header = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 4))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Texto Extraído",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        self._header_stats = ctk.CTkLabel(
            header, text="",
            font=ctk.CTkFont(size=12), text_color="gray",
        )
        self._header_stats.grid(row=0, column=1, sticky="e")

        # ── Área de texto (monoespaciada) ──
        self._textbox = ctk.CTkTextbox(
            self.main_frame,
            font=ctk.CTkFont(family="Consolas", size=13),
            wrap="none",
            corner_radius=8,
            state="disabled",
        )
        self._textbox.grid(row=1, column=0, sticky="nsew", padx=16, pady=4)

        # Texto placeholder
        self._set_text(
            "Selecciona una imagen y presiona «Procesar» para comenzar.\n\n"
            "El texto extraído aparecerá aquí con la estructura\n"
            "original del documento preservada."
        )

        # ── Panel de estadísticas ──
        stats_bar = ctk.CTkFrame(self.main_frame, height=60, corner_radius=8)
        stats_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=4)
        stats_bar.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._s_words = self._stat_card(stats_bar, "Palabras", "—", 0)
        self._s_conf = self._stat_card(stats_bar, "Confianza", "—", 1)
        self._s_blocks = self._stat_card(stats_bar, "Bloques", "—", 2)
        self._s_time = self._stat_card(stats_bar, "Tiempo", "—", 3)

        # ── Label archivos generados ──
        self._files_label = ctk.CTkLabel(
            self.main_frame, text="",
            font=ctk.CTkFont(size=11), text_color="gray", anchor="w",
        )
        self._files_label.grid(row=3, column=0, padx=22, pady=(2, 0), sticky="w")

        # ── Botones de acción ──
        actions = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        actions.grid(row=4, column=0, sticky="ew", padx=16, pady=(4, 12))

        self._copy_btn = ctk.CTkButton(
            actions, text="📋  Copiar texto",
            command=self._copy_text,
            height=36, width=150,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLOR_SUCCESS,
            hover_color=("#15803d", "#16a34a"),
            state="disabled",
        )
        self._copy_btn.pack(side="left", padx=(0, 8))

        self._open_folder_btn = ctk.CTkButton(
            actions, text="📂  Abrir carpeta",
            command=self._open_results_folder,
            height=36, width=145,
            font=ctk.CTkFont(size=12),
            fg_color="transparent", border_width=1,
            text_color=("gray10", "gray90"),
            state="disabled",
        )
        self._open_folder_btn.pack(side="left", padx=(0, 8))

        self._open_file_btn = ctk.CTkButton(
            actions, text="📄  Abrir archivo",
            command=self._open_first_file,
            height=36, width=145,
            font=ctk.CTkFont(size=12),
            fg_color="transparent", border_width=1,
            text_color=("gray10", "gray90"),
            state="disabled",
        )
        self._open_file_btn.pack(side="left")

    def _stat_card(self, parent, label: str, value: str, col: int) -> ctk.CTkLabel:
        """Crea una mini-tarjeta de estadística. Devuelve el label del valor."""
        cell = ctk.CTkFrame(parent, fg_color="transparent")
        cell.grid(row=0, column=col, padx=8, pady=8)

        ctk.CTkLabel(
            cell, text=label,
            font=ctk.CTkFont(size=10), text_color="gray",
        ).pack()

        val = ctk.CTkLabel(
            cell, text=value,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        val.pack()
        return val

    # ─────────────────────────────────────────
    # BARRA DE ESTADO
    # ─────────────────────────────────────────
    def _build_status_bar(self):
        """Construye la barra inferior con progreso y mensajes."""
        bar = ctk.CTkFrame(self, height=38, corner_radius=0)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        self._status_text = ctk.CTkLabel(
            bar, text="  Listo",
            font=ctk.CTkFont(size=12), text_color="gray",
        )
        self._status_text.grid(row=0, column=0, padx=14, pady=8, sticky="w")

        self._progress = ctk.CTkProgressBar(bar, height=6)
        self._progress.grid(row=0, column=1, padx=(0, 14), pady=8, sticky="ew")
        self._progress.set(0)

    # ─────────────────────────────────────────
    # Selección de imagen
    # ─────────────────────────────────────────
    def _browse_image(self):
        """Abre diálogo de selección de archivo de imagen."""
        path = filedialog.askopenfilename(
            title="Seleccionar imagen para OCR",
            filetypes=IMAGE_FILETYPES,
        )
        if path:
            self._load_image(path)

    def _paste_image_from_clipboard(self):
        """Pega una imagen del portapapeles del sistema.

        Intenta obtener la imagen con múltiples métodos según el SO:
          1. PIL.ImageGrab.grabclipboard() (Windows, macOS, Linux con
             wl-paste o xclip)
          2. Subproceso wl-paste (Wayland)
          3. Subproceso xclip (X11)

        La imagen se guarda como PNG temporal y se carga como imagen
        de entrada para el pipeline OCR.
        """
        if self._processing:
            return

        img = self._grab_clipboard_image()

        if img is None:
            messagebox.showinfo(
                "Sin imagen en portapapeles",
                "No se encontró ninguna imagen en el portapapeles.\n\n"
                "Copia una imagen (captura de pantalla, imagen de\n"
                "navegador, etc.) y vuelve a intentar.",
            )
            return

        # Guardar como PNG temporal
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", prefix="imageRD_clip_",
                delete=False,
            )
            img.save(tmp.name, format="PNG")
            tmp.close()

            tmp_path = Path(tmp.name)
            self._temp_files.append(tmp_path)

            self._load_image(str(tmp_path))
            self._update_status(
                f"Imagen pegada desde portapapeles ({img.size[0]}×{img.size[1]} px)"
            )
        except Exception as exc:
            messagebox.showerror(
                "Error al pegar imagen",
                f"No se pudo guardar la imagen del portapapeles:\n{exc}",
            )

    @staticmethod
    def _grab_clipboard_image() -> Optional[Image.Image]:
        """Obtiene imagen del portapapeles con múltiples estrategias.

        Returns:
            PIL.Image.Image si hay imagen, None en caso contrario.
        """
        from PIL import ImageGrab

        # ── Método 1: PIL ImageGrab (multiplataforma) ──
        try:
            clip = ImageGrab.grabclipboard()
            if isinstance(clip, Image.Image):
                return clip
            # En algunos casos devuelve lista de rutas de archivos
            if isinstance(clip, list) and clip:
                first = clip[0]
                if isinstance(first, str) and os.path.isfile(first):
                    return Image.open(first)
        except Exception:
            pass

        # ── Método 2: wl-paste (Linux / Wayland) ──
        try:
            result = subprocess.run(
                ["wl-paste", "--type", "image/png"],
                capture_output=True, timeout=3,
            )
            if result.returncode == 0 and result.stdout:
                return Image.open(io.BytesIO(result.stdout))
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        # ── Método 3: xclip (Linux / X11) ──
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard",
                 "-t", "image/png", "-o"],
                capture_output=True, timeout=3,
            )
            if result.returncode == 0 and result.stdout:
                return Image.open(io.BytesIO(result.stdout))
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        return None

    def _load_image(self, filepath: str):
        """Carga la imagen seleccionada y muestra el thumbnail."""
        self._image_path = filepath
        name = Path(filepath).name

        try:
            img = Image.open(filepath)
            w, h = img.size

            # Generar thumbnail proporcional
            max_w, max_h = 280, 110
            ratio = min(max_w / w, max_h / h)
            thumb_w, thumb_h = int(w * ratio), int(h * ratio)

            thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(
                light_image=thumb, dark_image=thumb,
                size=(thumb_w, thumb_h),
            )

            self._thumb_label.configure(image=ctk_img, text="")
            self._current_thumb = ctk_img  # evitar garbage collection

            self._img_info.configure(text=f"{name}  •  {w} × {h} px")

        except Exception:
            self._thumb_label.configure(image=None, text=f"📷  {name}")
            self._img_info.configure(text=name)

        self._update_status(f"Imagen cargada: {name}")

    # ─────────────────────────────────────────
    # Controles de la sidebar
    # ─────────────────────────────────────────
    def _on_workers_change(self, value: str):
        """Actualiza variable interna al cambiar selector de workers."""
        try:
            self._workers_var.set(int(value.split(" ")[0]))
        except (ValueError, IndexError):
            self._workers_var.set(0)

    def _toggle_theme(self):
        """Alterna entre modo oscuro y claro."""
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("light")
            self._theme_btn.configure(text="🌙  Cambiar a modo oscuro")
        else:
            ctk.set_appearance_mode("dark")
            self._theme_btn.configure(text="☀  Cambiar a modo claro")

    # ─────────────────────────────────────────
    # Procesamiento OCR (threaded)
    # ─────────────────────────────────────────
    def _start_processing(self):
        """Valida parámetros y lanza el OCR en un hilo separado."""
        if self._processing:
            return

        # Validar que haya imagen
        if not self._image_path:
            messagebox.showwarning(
                "Sin imagen",
                "Selecciona una imagen antes de procesar.",
            )
            return

        try:
            validate_image_path(self._image_path)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Error de imagen", str(exc))
            return

        # ── Bloquear UI ──
        self._processing = True
        self._process_btn.configure(state="disabled", text="⏳  Procesando…")
        self._browse_btn.configure(state="disabled")
        self._paste_btn.configure(state="disabled")
        self._copy_btn.configure(state="disabled")
        self._open_folder_btn.configure(state="disabled")
        self._open_file_btn.configure(state="disabled")

        self._progress.configure(mode="indeterminate")
        self._progress.start()

        self._set_text(
            "Procesando imagen, por favor espera…\n\n"
            "Esto puede tomar entre 5 y 30 segundos dependiendo\n"
            "del tamaño de la imagen y las opciones seleccionadas."
        )

        # ── Recopilar parámetros ──
        lang_key = self._lang_var.get()
        language = LANGUAGES.get(lang_key, lang_key)  # permite códigos directos

        params = dict(
            image_path=self._image_path,
            output_format=self._format_var.get(),
            language=language,
            min_confidence=0,  # siempre auto-adaptativa
            psm=3,
            preprocess=self._preproc_var.get(),
            multi_pass=self._multipass_var.get(),
            workers=self._workers_var.get(),
        )

        # ── Lanzar hilo ──
        threading.Thread(
            target=self._ocr_worker,
            args=(params,),
            daemon=True,
        ).start()

    def _ocr_worker(self, params: dict):
        """
        Ejecuta el pipeline OCR en segundo plano.

        Flujo: OCREngine → LayoutReconstructor → Exporters.
        Comunica resultados a la GUI vía self.after().
        """
        t0 = time.time()

        try:
            # Paso 1 — OCR
            self.after(0, self._update_status, "Iniciando motor OCR…")
            engine = OCREngine(
                language=params["language"],
                min_confidence=params["min_confidence"],
                psm=params["psm"],
                multi_pass=params["multi_pass"],
                workers=params["workers"],
                auto_psm=True,
            )
            ocr_result: OCRResult = engine.extract(
                params["image_path"],
                preprocess=params["preprocess"],
            )

            if ocr_result.word_count == 0:
                self.after(
                    0, self._on_ocr_error,
                    "No se detectó texto en la imagen.\n\n"
                    "Sugerencias:\n"
                    "  • Verifica que la imagen contenga texto legible.\n"
                    "  • Desactiva el preprocesamiento.\n"
                    "  • Verifica que el idioma sea correcto.",
                )
                return

            # Paso 2 — Reconstrucción de layout
            self.after(0, self._update_status, "Reconstruyendo estructura…")
            reconstructor = LayoutReconstructor()
            reconstructed: ReconstructedDocument = reconstructor.reconstruct(
                ocr_result,
            )

            # Paso 3 — Exportación
            # Forzar guardado en resultados/ del directorio del proyecto
            self.after(0, self._update_status, "Exportando resultados…")
            _project_results = Path(__file__).resolve().parent / "resultados"
            _project_results.mkdir(parents=True, exist_ok=True)

            from utils import generate_output_path

            fmt = params["output_format"]
            if fmt == "all":
                # Para cada formato, generar ruta en resultados/ del proyecto
                generated: List[Path] = []
                for sub_fmt in ("txt", "json", "docx"):
                    _out = generate_output_path(
                        Path(params["image_path"]), sub_fmt,
                        output_dir=str(_project_results),
                    )
                    sub_files = export_all(
                        ocr_result=ocr_result,
                        reconstructed=reconstructed,
                        output_format=sub_fmt,
                        input_path=params["image_path"],
                        output_path=str(_out),
                    )
                    generated.extend(sub_files)
            else:
                _out_path = generate_output_path(
                    Path(params["image_path"]), fmt,
                    output_dir=str(_project_results),
                )
                generated = export_all(
                    ocr_result=ocr_result,
                    reconstructed=reconstructed,
                    output_format=fmt,
                    input_path=params["image_path"],
                    output_path=str(_out_path),
                )

            elapsed = time.time() - t0

            # Enviar resultados a la GUI
            self.after(
                0, self._on_ocr_complete,
                ocr_result, reconstructed, generated, elapsed,
            )

        except Exception as exc:
            self.after(
                0, self._on_ocr_error,
                f"Error durante el procesamiento:\n\n{exc}",
            )

    # ─────────────────────────────────────────
    # Callbacks de resultado
    # ─────────────────────────────────────────
    def _on_ocr_complete(
        self,
        ocr_result: OCRResult,
        reconstructed: ReconstructedDocument,
        generated: List[Path],
        elapsed: float,
    ):
        """Actualiza toda la interfaz con los resultados del OCR."""
        self._ocr_result = ocr_result
        self._reconstructed = reconstructed
        self._generated_files = generated

        # Texto extraído
        self._set_text(reconstructed.formatted_text)

        # Estadísticas
        self._s_words.configure(text=str(ocr_result.word_count))
        self._s_conf.configure(text=f"{ocr_result.avg_confidence:.1f}%")
        self._s_blocks.configure(text=str(len(ocr_result.blocks)))
        self._s_time.configure(text=f"{elapsed:.1f}s")

        self._header_stats.configure(
            text=f"{ocr_result.word_count} palabras  •  "
                 f"{ocr_result.avg_confidence:.1f}% confianza",
        )

        # Archivos generados
        names = "   ".join(f"✓ {f.name}" for f in generated)
        self._files_label.configure(text=f"Archivos:  {names}")

        # Habilitar acciones
        self._copy_btn.configure(state="normal")
        self._open_folder_btn.configure(state="normal")
        if generated:
            self._open_file_btn.configure(state="normal")

        # Restaurar UI
        self._finish_processing()
        self._update_status(
            f"✓ Completado — {ocr_result.word_count} palabras, "
            f"{ocr_result.avg_confidence:.1f}% confianza, {elapsed:.1f}s"
        )

    def _on_ocr_error(self, message: str):
        """Muestra el error y restaura la interfaz."""
        self._finish_processing()
        self._set_text(f"⚠  {message}")
        self._update_status("✗ Error en el procesamiento")
        messagebox.showerror("Error de OCR", message)

    def _finish_processing(self):
        """Restaura controles tras procesamiento (éxito o error)."""
        self._processing = False
        self._process_btn.configure(
            state="normal", text="🔍  PROCESAR IMAGEN",
        )
        self._browse_btn.configure(state="normal")
        self._paste_btn.configure(state="normal")
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(1.0)

    # ─────────────────────────────────────────
    # Acciones sobre el resultado
    # ─────────────────────────────────────────
    def _copy_text(self):
        """Copia el texto extraído al portapapeles del sistema."""
        if not self._reconstructed or not self._reconstructed.formatted_text:
            return

        self.clipboard_clear()
        self.clipboard_append(self._reconstructed.formatted_text)

        # Feedback visual breve
        self._copy_btn.configure(text="✓  ¡Copiado!", fg_color=("#15803d", "#166534"))
        self.after(
            2000,
            lambda: self._copy_btn.configure(
                text="📋  Copiar texto", fg_color=COLOR_SUCCESS,
            ),
        )

    def _open_results_folder(self):
        """Abre la carpeta de resultados en el explorador de archivos."""
        folder = None
        if self._generated_files:
            folder = self._generated_files[0].parent
        elif self._image_path:
            folder = Path(self._image_path).parent / "resultados"

        if folder and folder.exists():
            self._open_path(str(folder))

    def _open_first_file(self):
        """Abre el primer archivo generado con la aplicación por defecto."""
        if self._generated_files and self._generated_files[0].exists():
            self._open_path(str(self._generated_files[0]))

    @staticmethod
    def _open_path(path: str):
        """Abre un archivo o carpeta de forma multiplataforma."""
        system = platform.system()
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    # ─────────────────────────────────────────
    # Utilidades de UI
    # ─────────────────────────────────────────
    def _set_text(self, content: str):
        """Reemplaza el contenido del área de texto."""
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", content)
        self._textbox.configure(state="disabled")

    def _update_status(self, text: str):
        """Actualiza el mensaje de la barra de estado."""
        self._status_text.configure(text=f"  {text}")

    def _on_log_message(self, message: str):
        """Recibe mensajes del logger y los muestra en la barra de estado."""
        try:
            self.after(0, self._update_status, message)
        except Exception:
            pass

    def _check_tesseract(self):
        """
        Verifica que Tesseract esté disponible al iniciar la GUI.

        Si no se encuentra, ofrece descargarlo automáticamente
        usando tesseract_manager.
        """
        import shutil as _shutil

        # Verificar si está bundled o en el sistema
        try:
            from tesseract_manager import is_bundled, get_tesseract_path
            if is_bundled():
                return  # Todo bien, está integrado
        except ImportError:
            pass

        if _shutil.which("tesseract"):
            return  # Está en el sistema

        # No se encontró — preguntar al usuario
        respuesta = messagebox.askyesno(
            "Tesseract OCR no encontrado",
            "ImageRD necesita Tesseract OCR para funcionar.\n\n"
            "¿Deseas descargarlo automáticamente?\n"
            "(Se guardará dentro del proyecto, ~30-50 MB)\n\n"
            "Requiere conexión a internet.",
        )

        if respuesta:
            self._setup_tesseract_gui()
        else:
            self._update_status("⚠ Tesseract no instalado — el procesamiento OCR no funcionará")

    def _setup_tesseract_gui(self):
        """
        Descarga e instala Tesseract en segundo plano desde la GUI.

        Muestra progreso en la barra de estado y bloquea el botón
        de procesar durante la instalación.
        """
        self._process_btn.configure(state="disabled", text="⏳  Instalando Tesseract…")
        self._progress.configure(mode="indeterminate")
        self._progress.start()
        self._update_status("Descargando Tesseract OCR...")

        def _worker():
            try:
                from tesseract_manager import setup as tess_setup
                tess_setup(languages=["eng", "spa", "osd"])
                self.after(0, self._on_tesseract_installed, True, "")
            except Exception as exc:
                self.after(0, self._on_tesseract_installed, False, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_tesseract_installed(self, success: bool, error: str):
        """Callback cuando la instalación de Tesseract termina."""
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._process_btn.configure(state="normal", text="▶  Procesar imagen")

        if success:
            self._update_status("✓ Tesseract instalado correctamente")
            messagebox.showinfo(
                "Instalación completada",
                "Tesseract OCR se instaló correctamente.\n"
                "Ya puedes procesar imágenes.",
            )
        else:
            self._update_status(f"✗ Error instalando Tesseract: {error}")
            messagebox.showerror(
                "Error de instalación",
                f"No se pudo instalar Tesseract:\n{error}\n\n"
                "Puedes instalarlo manualmente:\n"
                "  sudo pacman -S tesseract  (Arch)\n"
                "  sudo apt install tesseract-ocr  (Debian/Ubuntu)",
            )

    def _on_close(self):
        """Maneja el cierre de la ventana."""
        if self._processing:
            if not messagebox.askyesno(
                "Procesamiento en curso",
                "Hay un procesamiento OCR en curso.\n¿Deseas cerrar de todas formas?",
            ):
                return

        # Limpiar archivos temporales del portapapeles
        for tmp in self._temp_files:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

        # Limpiar log handler
        logger.removeHandler(self._log_handler)
        self.destroy()


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────
def main():
    """Inicia la interfaz gráfica de ImageRD."""
    app = ImageRDApp()
    app.mainloop()


if __name__ == "__main__":
    main()
