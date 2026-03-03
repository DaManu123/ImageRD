"""
tesseract_manager.py — Gestor de Tesseract OCR integrado.

Descarga, configura y mantiene una instalación local de Tesseract OCR
dentro del proyecto (vendor/tesseract/), eliminando la necesidad de
instalar Tesseract como paquete del sistema.

Fuentes de descarga:
  - Binarios Linux/macOS: conda-forge (paquetes autocontenidos)
  - Binarios Windows: UB-Mannheim GitHub releases
  - Datos de idioma (tessdata): tesseract-ocr/tessdata_fast en GitHub

Uso:
    from tesseract_manager import ensure_tesseract, get_tesseract_path

    # Asegura que Tesseract esté disponible (descarga si es necesario)
    tesseract_bin = ensure_tesseract()

    # Solo obtener la ruta (sin descargar)
    path = get_tesseract_path()
"""

import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# Rutas del proyecto
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

VENDOR_DIR = PROJECT_ROOT / "vendor" / "tesseract"
BIN_DIR = VENDOR_DIR / "bin"
LIB_DIR = VENDOR_DIR / "lib"
TESSDATA_DIR = VENDOR_DIR / "share" / "tessdata"
CACHE_DIR = VENDOR_DIR / ".cache"


# ─────────────────────────────────────────────
# URLs y configuración de descarga
# ─────────────────────────────────────────────
CONDA_FORGE_API = "https://api.anaconda.org/package/conda-forge"
CONDA_FORGE_DL = "https://conda.anaconda.org/conda-forge"

TESSDATA_GITHUB_URL = (
    "https://github.com/tesseract-ocr/tessdata_fast/raw/main"
)

# Idiomas descargados por defecto
DEFAULT_LANGUAGES = ["eng", "spa", "osd"]

# Paquetes conda-forge necesarios para cada plataforma.
# Incluye todas las dependencias críticas para que el binario funcione standalone.
# Formato: (nombre_paquete, prefijo_versión_o_None)
#   - None = versión más reciente
#   - "1.1" = la más reciente que empiece con "1.1"
_CONDA_PACKAGES: Dict[str, List[tuple]] = {
    "linux-64": [
        ("tesseract", None),
        ("leptonica", None),
        ("libarchive", None),
        ("jpeg", None),
        ("libtiff", None),
        ("libxml2", None),
        ("lerc", None),
        ("icu", None),
        ("libiconv", None),
        ("openssl", "1.1"),  # libarchive requiere OpenSSL 1.1.x
    ],
    "linux-aarch64": [
        ("tesseract", None),
        ("leptonica", None),
        ("libarchive", None),
        ("jpeg", None),
        ("libtiff", None),
        ("libxml2", None),
        ("lerc", None),
        ("icu", None),
        ("libiconv", None),
        ("openssl", "1.1"),
    ],
    "osx-64": [
        ("tesseract", None),
        ("leptonica", None),
        ("libarchive", None),
        ("jpeg", None),
        ("libtiff", None),
        ("libxml2", None),
        ("lerc", None),
        ("icu", None),
        ("libiconv", None),
        ("openssl", "1.1"),
    ],
    "osx-arm64": [
        ("tesseract", None),
        ("leptonica", None),
        ("libarchive", None),
        ("jpeg", None),
        ("libtiff", None),
        ("libxml2", None),
        ("lerc", None),
        ("icu", None),
        ("libiconv", None),
        ("openssl", "1.1"),
    ],
}

# UB-Mannheim para Windows
UB_MANNHEIM_API = (
    "https://api.github.com/repos/UB-Mannheim/tesseract/releases/latest"
)


# ─────────────────────────────────────────────
# Utilidades internas de salida
# ─────────────────────────────────────────────
def _print(msg: str) -> None:
    """Imprime un mensaje con prefijo del gestor."""
    print(f"  │ {msg}")


def _print_header(msg: str) -> None:
    """Imprime un encabezado decorativo."""
    print(f"\n  ┌{'─' * 52}┐")
    print(f"  │  {msg:<50} │")
    print(f"  └{'─' * 52}┘")


def _print_step(step: int, total: int, msg: str) -> None:
    """Imprime un paso numerado."""
    print(f"  ▸ [{step}/{total}] {msg}")


def _print_ok(msg: str) -> None:
    """Imprime un mensaje de éxito."""
    print(f"  ✓ {msg}")


def _print_warn(msg: str) -> None:
    """Imprime una advertencia."""
    print(f"  ⚠ {msg}")


def _print_error(msg: str) -> None:
    """Imprime un error."""
    print(f"  ✗ {msg}")


# ─────────────────────────────────────────────
# Detección de plataforma
# ─────────────────────────────────────────────
def _get_conda_platform() -> str:
    """
    Determina el identificador de plataforma de conda-forge.

    Returns:
        Cadena como 'linux-64', 'osx-arm64', 'win-64', etc.

    Raises:
        OSError: Si la plataforma no está soportada.
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux-64"
        elif machine in ("aarch64", "arm64"):
            return "linux-aarch64"
    elif system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "osx-arm64"
        else:
            return "osx-64"
    elif system == "windows":
        return "win-64"

    raise OSError(
        f"Plataforma no soportada: {system}/{machine}.\n"
        "Plataformas soportadas: Linux x64/arm64, macOS x64/arm64, Windows x64."
    )


# ─────────────────────────────────────────────
# Descargas
# ─────────────────────────────────────────────
def _download_with_progress(
    url: str,
    dest: Path,
    label: str = "",
) -> Path:
    """
    Descarga un archivo con barra de progreso en consola.

    Args:
        url: URL del archivo a descargar.
        dest: Ruta de destino (archivo o directorio).
        label: Etiqueta descriptiva para la barra de progreso.

    Returns:
        Ruta al archivo descargado.
    """
    if dest.is_dir():
        filename = url.split("/")[-1].split("?")[0]
        dest = dest / filename

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not label:
        label = dest.name

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ImageRD-TesseractManager/1.0")

        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 65536
            data = bytearray()

            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                data.extend(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = downloaded * 100 // total
                    mb_dl = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    bar_len = 25
                    filled = int(bar_len * pct // 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(
                        f"\r    {label}: {bar} {pct:>3}% "
                        f"({mb_dl:.1f}/{mb_total:.1f} MB)",
                        end="",
                        flush=True,
                    )

            print()  # Nueva línea después de la barra

        dest.write_bytes(data)
        return dest

    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Error descargando {label}: {exc}"
        ) from exc


def _fetch_json(url: str) -> dict:
    """
    Descarga y parsea un JSON desde una URL.

    Args:
        url: URL del recurso JSON.

    Returns:
        Diccionario con los datos parseados.
    """
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ImageRD-TesseractManager/1.0")

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


# ─────────────────────────────────────────────
# Extracción de paquetes
# ─────────────────────────────────────────────
def _extract_tar_bz2(filepath: Path, target_dir: Path) -> None:
    """
    Extrae un paquete conda .tar.bz2 al directorio destino.

    Excluye el directorio info/ (metadatos de conda).

    Args:
        filepath: Ruta al archivo .tar.bz2
        target_dir: Directorio donde extraer los archivos.
    """
    with tarfile.open(filepath, "r:bz2") as tar:
        members = [
            m for m in tar.getmembers()
            if not m.name.startswith("info/") and not m.name.startswith("info\\")
        ]
        tar.extractall(path=target_dir, members=members)


def _extract_conda_pkg(filepath: Path, target_dir: Path) -> None:
    """
    Extrae un paquete .conda (formato ZIP con tar.zst dentro).

    Requiere el comando 'zstd' en el sistema para descomprimir.

    Args:
        filepath: Ruta al archivo .conda
        target_dir: Directorio donde extraer los archivos.

    Raises:
        RuntimeError: Si zstd no está disponible o la extracción falla.
    """
    if not shutil.which("zstd"):
        raise RuntimeError(
            "El comando 'zstd' es necesario para extraer paquetes .conda.\n"
            "Instálalo con: sudo pacman -S zstd (Arch) o sudo apt install zstd (Debian)."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Extraer el ZIP
        with zipfile.ZipFile(filepath) as zf:
            zf.extractall(tmp)

        # Buscar el tar.zst de datos del paquete
        pkg_files = list(tmp.glob("pkg-*.tar.zst"))
        if not pkg_files:
            raise RuntimeError(
                f"Formato .conda no válido: no se encontró pkg-*.tar.zst en {filepath.name}"
            )

        pkg_zst = pkg_files[0]
        pkg_tar = tmp / pkg_zst.name.replace(".tar.zst", ".tar")

        # Descomprimir zst → tar
        result = subprocess.run(
            ["zstd", "-d", str(pkg_zst), "-o", str(pkg_tar)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Error descomprimiendo {pkg_zst.name}: {result.stderr}")

        # Extraer tar
        with tarfile.open(pkg_tar) as tar:
            members = [
                m for m in tar.getmembers()
                if not m.name.startswith("info/")
                and not m.name.startswith("info\\")
            ]
            tar.extractall(path=target_dir, members=members)


def _extract_package(filepath: Path, target_dir: Path) -> None:
    """
    Extrae un paquete conda (.tar.bz2 o .conda) al directorio destino.

    Args:
        filepath: Ruta al paquete descargado.
        target_dir: Directorio donde extraer.
    """
    name = filepath.name.lower()

    if name.endswith(".tar.bz2"):
        _extract_tar_bz2(filepath, target_dir)
    elif name.endswith(".conda"):
        _extract_conda_pkg(filepath, target_dir)
    else:
        raise ValueError(f"Formato de paquete no soportado: {filepath.name}")


# ─────────────────────────────────────────────
# Resolución de paquetes conda-forge
# ─────────────────────────────────────────────
def _get_conda_package_url(
    package_name: str,
    plat: str,
    version_prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Obtiene la URL de descarga de un paquete conda-forge.

    Consulta la API de Anaconda para encontrar la versión más reciente
    del paquete para la plataforma especificada.

    Args:
        package_name: Nombre del paquete (e.g., 'tesseract').
        plat: Plataforma conda (e.g., 'linux-64').
        version_prefix: Prefijo de versión para filtrar (e.g., '1.1' para OpenSSL 1.1.x).

    Returns:
        Tupla (url_descarga, nombre_archivo).

    Raises:
        RuntimeError: Si el paquete no se encuentra.
    """
    api_url = f"{CONDA_FORGE_API}/{package_name}/files"

    try:
        files = _fetch_json(api_url)
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo consultar conda-forge para '{package_name}': {exc}"
        ) from exc

    # Filtrar por plataforma — preferir .tar.bz2 (soporte nativo Python)
    tar_bz2 = []
    conda_fmt = []

    for f in files:
        subdir = f.get("attrs", {}).get("subdir", "")
        if subdir != plat:
            continue

        # Filtrar por prefijo de versión si se especificó
        if version_prefix is not None:
            ver = f.get("version", "")
            if not ver.startswith(version_prefix):
                continue

        basename = f.get("basename", "")
        if basename.endswith(".tar.bz2"):
            tar_bz2.append(f)
        elif basename.endswith(".conda"):
            conda_fmt.append(f)

    # Preferir .tar.bz2 sobre .conda
    candidates = tar_bz2 if tar_bz2 else conda_fmt

    if not candidates:
        raise RuntimeError(
            f"No se encontró el paquete '{package_name}' para plataforma '{plat}' "
            f"en conda-forge."
        )

    # Ordenar por versión (más reciente primero) + fecha de subida como desempate
    def _version_key(entry):
        ver = entry.get("version", "0")
        parts = []
        for p in re.split(r"[.\-_]", ver):
            # Manejar versiones como "9e" → (9, 5) donde e=5
            match = re.match(r"^(\d+)([a-z]?)$", p)
            if match:
                parts.append(int(match.group(1)))
                if match.group(2):
                    parts.append(ord(match.group(2)) - ord("a") + 1)
            else:
                try:
                    parts.append(int(p))
                except ValueError:
                    parts.append(0)
        return parts

    candidates.sort(
        key=lambda f: (_version_key(f), f.get("upload_time", "")),
        reverse=True,
    )

    best = candidates[0]
    dl_url = best.get("download_url", "")

    # La URL puede empezar con // (protocolo relativo)
    if dl_url.startswith("//"):
        dl_url = f"https:{dl_url}"
    elif not dl_url.startswith("http"):
        dl_url = f"{CONDA_FORGE_DL}/{plat}/{best['basename']}"

    return dl_url, best["basename"]


# ─────────────────────────────────────────────
# Setup por plataforma
# ─────────────────────────────────────────────
def _setup_unix(plat: str) -> None:
    """
    Descarga e instala Tesseract para Linux o macOS desde conda-forge.

    Descarga los paquetes necesarios (tesseract, leptonica, libarchive),
    los extrae a vendor/tesseract/ y configura los permisos.

    Args:
        plat: Plataforma conda (e.g., 'linux-64', 'osx-arm64').
    """
    packages = _CONDA_PACKAGES.get(plat, [])
    if not packages:
        raise OSError(f"No hay paquetes configurados para la plataforma: {plat}")

    total_steps = len(packages)

    for i, (pkg_name, ver_prefix) in enumerate(packages, 1):
        _print_step(i, total_steps, f"Descargando {pkg_name}...")

        try:
            url, filename = _get_conda_package_url(pkg_name, plat, ver_prefix)
        except RuntimeError as exc:
            _print_warn(f"No se pudo obtener {pkg_name}: {exc}")
            continue

        # Descargar
        pkg_file = _download_with_progress(url, CACHE_DIR, label=pkg_name)

        # Extraer
        _print(f"Extrayendo {filename}...")
        _extract_package(pkg_file, VENDOR_DIR)
        _print_ok(f"{pkg_name} instalado.")

    # Hacer ejecutable el binario
    binary = BIN_DIR / "tesseract"
    if binary.exists():
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # También hacer ejecutables todas las librerías .so
    if LIB_DIR.exists():
        for so_file in LIB_DIR.glob("*.so*"):
            so_file.chmod(so_file.stat().st_mode | stat.S_IEXEC)


def _setup_windows() -> None:
    """
    Descarga e instala Tesseract para Windows desde UB-Mannheim.

    Descarga el instalador .exe, lo ejecuta en modo silencioso extrayendo
    al directorio vendor/tesseract/, o descarga desde conda-forge como
    alternativa.
    """
    # Intentar primero con conda-forge (más limpio)
    plat = "win-64"
    packages = ["tesseract", "leptonica"]
    total_steps = len(packages)

    for i, pkg_name in enumerate(packages, 1):
        _print_step(i, total_steps, f"Descargando {pkg_name}...")

        try:
            url, filename = _get_conda_package_url(pkg_name, plat)
        except RuntimeError as exc:
            _print_warn(f"No se pudo obtener {pkg_name}: {exc}")
            continue

        pkg_file = _download_with_progress(url, CACHE_DIR, label=pkg_name)
        _print(f"Extrayendo {filename}...")
        _extract_package(pkg_file, VENDOR_DIR)
        _print_ok(f"{pkg_name} instalado.")

    # En Windows, el binario podría estar en Library/bin/ según convención conda
    win_bin = VENDOR_DIR / "Library" / "bin" / "tesseract.exe"
    if win_bin.exists() and not (BIN_DIR / "tesseract.exe").exists():
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(win_bin, BIN_DIR / "tesseract.exe")

    # También copiar DLLs de Library/bin/ a bin/
    win_lib = VENDOR_DIR / "Library" / "bin"
    if win_lib.exists():
        for dll in win_lib.glob("*.dll"):
            dest = BIN_DIR / dll.name
            if not dest.exists():
                shutil.copy2(dll, dest)


# ─────────────────────────────────────────────
# Descarga de tessdata (datos de idioma)
# ─────────────────────────────────────────────
def _download_tessdata(languages: List[str]) -> None:
    """
    Descarga archivos de datos de idioma (.traineddata) desde GitHub.

    Los archivos se descargan desde el repositorio
    tesseract-ocr/tessdata_fast (modelos rápidos, ~1-4 MB cada uno).

    Args:
        languages: Lista de códigos de idioma (e.g., ['eng', 'spa', 'osd']).
    """
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        dest = TESSDATA_DIR / f"{lang}.traineddata"
        if dest.exists():
            _print_ok(f"tessdata/{lang}.traineddata ya existe.")
            continue

        url = f"{TESSDATA_GITHUB_URL}/{lang}.traineddata"
        _print(f"Descargando datos de idioma: {lang}...")

        try:
            _download_with_progress(url, dest, label=f"tessdata/{lang}")
            _print_ok(f"{lang}.traineddata descargado.")
        except ConnectionError as exc:
            _print_warn(
                f"No se pudo descargar {lang}.traineddata: {exc}\n"
                f"   Verifica que el código de idioma '{lang}' sea correcto."
            )


# ─────────────────────────────────────────────
# Configuración del entorno de ejecución
# ─────────────────────────────────────────────
def configure_environment() -> None:
    """
    Configura las variables de entorno necesarias para que pytesseract
    encuentre el Tesseract bundled y sus dependencias.

    Establece:
      - LD_LIBRARY_PATH (Linux/macOS): para encontrar librerías .so/.dylib
      - TESSDATA_PREFIX: para encontrar los archivos .traineddata
      - PATH (Windows): para encontrar DLLs
    """
    if LIB_DIR.exists():
        lib_str = str(LIB_DIR)
        system = platform.system().lower()

        if system in ("linux", "darwin"):
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_str not in existing:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{lib_str}:{existing}" if existing else lib_str
                )
        elif system == "windows":
            existing = os.environ.get("PATH", "")
            bin_str = str(BIN_DIR)
            if bin_str not in existing:
                os.environ["PATH"] = f"{bin_str};{lib_str};{existing}"

    if TESSDATA_DIR.exists():
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


# ─────────────────────────────────────────────
# Verificación de la instalación
# ─────────────────────────────────────────────
def _verify_installation() -> bool:
    """
    Verifica que el Tesseract bundled funcione correctamente.

    Ejecuta `tesseract --version` y comprueba que retorne exitosamente.

    Returns:
        True si la verificación es exitosa, False en caso contrario.
    """
    binary = get_tesseract_path()
    if binary is None:
        return False

    # Configurar entorno antes de ejecutar
    configure_environment()

    env = os.environ.copy()

    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _diagnose_missing_libs() -> List[str]:
    """
    Diagnostica librerías compartidas faltantes del binario de Tesseract.

    Usa 'ldd' (Linux) para identificar dependencias no resueltas.

    Returns:
        Lista de nombres de librerías faltantes.
    """
    binary = get_tesseract_path()
    if binary is None or platform.system().lower() != "linux":
        return []

    configure_environment()
    env = os.environ.copy()

    try:
        result = subprocess.run(
            ["ldd", str(binary)],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        missing = []
        for line in result.stdout.splitlines():
            if "not found" in line:
                lib_name = line.strip().split()[0]
                missing.append(lib_name)
        return missing
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# ─────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────
def is_bundled() -> bool:
    """
    Verifica si existe una instalación local de Tesseract en vendor/.

    Returns:
        True si el binario de Tesseract existe en el directorio vendor.
    """
    path = get_tesseract_path()
    return path is not None and path.exists()


def get_tesseract_path() -> Optional[Path]:
    """
    Obtiene la ruta al ejecutable de Tesseract bundled.

    Returns:
        Path al binario, o None si no existe.
    """
    if sys.platform == "win32":
        binary = BIN_DIR / "tesseract.exe"
    else:
        binary = BIN_DIR / "tesseract"

    return binary if binary.exists() else None


def get_tessdata_path() -> Optional[Path]:
    """
    Obtiene la ruta al directorio tessdata del Tesseract bundled.

    Returns:
        Path al directorio tessdata, o None si no existe.
    """
    return TESSDATA_DIR if TESSDATA_DIR.exists() else None


def setup(
    languages: Optional[List[str]] = None,
    force: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Instala Tesseract OCR localmente en el proyecto.

    Descarga los binarios pre-compilados para la plataforma actual
    desde conda-forge, extrae las librerías y datos de idioma necesarios.

    Args:
        languages: Idiomas a descargar. Default: ['eng', 'spa', 'osd'].
        force: Si True, reinstala aunque ya exista una instalación.
        progress_callback: Función opcional para reportar progreso.

    Returns:
        Path al ejecutable de Tesseract instalado.

    Raises:
        OSError: Si la plataforma no está soportada.
        RuntimeError: Si la descarga o instalación falla.
        ConnectionError: Si no hay conexión a internet.
    """
    if is_bundled() and not force:
        _print_ok("Tesseract ya está instalado localmente.")
        configure_environment()
        return get_tesseract_path()

    _print_header("Instalando Tesseract OCR integrado")

    # Limpiar instalación previa si se fuerza
    if force and VENDOR_DIR.exists():
        _print("Eliminando instalación anterior...")
        shutil.rmtree(VENDOR_DIR)

    # Crear directorios
    for d in (VENDOR_DIR, BIN_DIR, LIB_DIR, TESSDATA_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Detectar plataforma
    plat = _get_conda_platform()
    _print(f"Plataforma detectada: {plat}")

    # Descargar e instalar binarios
    _print("")
    _print("Paso 1: Descargando binarios de Tesseract...")
    _print("")

    if plat == "win-64":
        _setup_windows()
    else:
        _setup_unix(plat)

    # Descargar tessdata
    _print("")
    _print("Paso 2: Descargando datos de idioma...")
    _print("")

    langs = languages or DEFAULT_LANGUAGES
    _download_tessdata(langs)

    # Configurar entorno
    configure_environment()

    # Verificar instalación
    _print("")
    _print("Paso 3: Verificando instalación...")

    if _verify_installation():
        binary = get_tesseract_path()
        # Obtener versión
        env = os.environ.copy()
        try:
            result = subprocess.run(
                [str(binary), "--version"],
                capture_output=True,
                text=True,
                env=env,
                timeout=10,
            )
            version_line = result.stdout.strip().split("\n")[0] if result.stdout else "desconocida"
        except Exception:
            version_line = "desconocida"

        _print_ok(f"Tesseract instalado correctamente: {version_line}")
        _print_ok(f"Ubicación: {binary}")
        _print_ok(f"Tessdata: {TESSDATA_DIR}")

        # Limpiar caché de descargas
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)

        return binary

    else:
        # Diagnóstico de errores
        missing = _diagnose_missing_libs()
        if missing:
            _print_error("Librerías faltantes detectadas:")
            for lib in missing:
                _print_error(f"  → {lib}")
            _print("")
            _print("Intentando descargar dependencias adicionales...")

            # Intentar descargar dependencias adicionales
            _try_resolve_missing_libs(missing, plat)

            # Re-verificar
            if _verify_installation():
                binary = get_tesseract_path()
                _print_ok("Tesseract instalado correctamente tras resolver dependencias.")
                if CACHE_DIR.exists():
                    shutil.rmtree(CACHE_DIR)
                return binary

        _print_error(
            "No se pudo verificar la instalación de Tesseract.\n"
            "  Esto puede deberse a librerías del sistema faltantes.\n"
            "  Intenta instalar: sudo pacman -S tesseract (Arch) o\n"
            "                    sudo apt install tesseract-ocr (Debian/Ubuntu)"
        )

        # Limpiar caché
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)

        # Aún retornar la ruta, puede que funcione en runtime
        return get_tesseract_path()


def _try_resolve_missing_libs(missing: List[str], plat: str) -> None:
    """
    Intenta resolver librerías faltantes descargando paquetes adicionales.

    Mapea nombres de librerías a paquetes conda-forge y descarga
    los que falten.

    Args:
        missing: Lista de nombres de librerías faltantes.
        plat: Plataforma conda.
    """
    # Mapeo de nombres de librería a paquetes conda-forge
    lib_to_package = {
        "libarchive": "libarchive",
        "libcurl": "libcurl",
        "libnghttp2": "libnghttp2",
        "libssh2": "libssh2",
        "libzstd": "zstd",
        "liblzma": "xz",
        "libicu": "icu",
        "libpng": "libpng",
        "libtiff": "libtiff",
        "libjpeg": "jpeg",
        "libwebp": "libwebp",
        "libgif": "giflib",
        "libopenjp2": "openjpeg",
        "libxml2": "libxml2",
        "libLerc": "lerc",
        "liblerc": "lerc",
        "libiconv": "libiconv",
        "libicuuc": "icu",
        "libicui18n": "icu",
        "libicudata": "icu",
        "libcrypto": "openssl",
        "libssl": "openssl",
    }

    packages_to_try = set()
    for lib in missing:
        for pattern, pkg in lib_to_package.items():
            if pattern in lib:
                packages_to_try.add(pkg)
                break

    for pkg_name in packages_to_try:
        try:
            _print(f"Descargando dependencia: {pkg_name}...")
            url, filename = _get_conda_package_url(pkg_name, plat)
            pkg_file = _download_with_progress(url, CACHE_DIR, label=pkg_name)
            _extract_package(pkg_file, VENDOR_DIR)
            _print_ok(f"{pkg_name} instalado.")
        except Exception as exc:
            _print_warn(f"No se pudo descargar {pkg_name}: {exc}")


def ensure_tesseract(
    languages: Optional[List[str]] = None,
    interactive: bool = True,
) -> Optional[str]:
    """
    Asegura que Tesseract esté disponible, descargándolo si es necesario.

    Esta es la función principal que debe llamarse al iniciar el programa.
    Si Tesseract está bundled, lo configura. Si no, ofrece descargarlo.

    Args:
        languages: Idiomas adicionales a descargar.
        interactive: Si True, pregunta antes de descargar.

    Returns:
        Ruta al ejecutable de Tesseract, o None si no está disponible.
    """
    # 1. Verificar si ya está bundled
    if is_bundled():
        configure_environment()
        return str(get_tesseract_path())

    # 2. Verificar si está en el sistema
    system_tesseract = shutil.which("tesseract")
    if system_tesseract:
        return system_tesseract

    # 3. No está en ningún lado — ofrecer descarga
    if interactive:
        print("\n" + "═" * 55)
        print("  Tesseract OCR no encontrado")
        print("═" * 55)
        print()
        print("  ImageRD puede descargar Tesseract automáticamente")
        print("  y guardarlo dentro del proyecto (~30-50 MB).")
        print()

        try:
            respuesta = input("  ¿Descargar Tesseract ahora? [S/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if respuesta in ("", "s", "si", "sí", "y", "yes"):
            langs = languages or DEFAULT_LANGUAGES
            try:
                binary = setup(languages=langs)
                return str(binary) if binary else None
            except Exception as exc:
                _print_error(f"Error durante la instalación: {exc}")
                return None
        else:
            _print_warn(
                "Tesseract no instalado. El programa no podrá procesar imágenes.\n"
                "  Opciones:\n"
                "    - Ejecuta: python -m tesseract_manager\n"
                "    - O instala manualmente: sudo pacman -S tesseract"
            )
            return None

    else:
        # Modo no interactivo: intentar descargar directamente
        try:
            langs = languages or DEFAULT_LANGUAGES
            binary = setup(languages=langs)
            return str(binary) if binary else None
        except Exception:
            return None


def add_language(lang_code: str) -> bool:
    """
    Descarga un idioma adicional para Tesseract.

    Args:
        lang_code: Código del idioma (e.g., 'fra', 'deu', 'por').

    Returns:
        True si la descarga fue exitosa.
    """
    dest = TESSDATA_DIR / f"{lang_code}.traineddata"
    if dest.exists():
        _print_ok(f"El idioma '{lang_code}' ya está instalado.")
        return True

    url = f"{TESSDATA_GITHUB_URL}/{lang_code}.traineddata"
    _print(f"Descargando idioma: {lang_code}...")

    try:
        _download_with_progress(url, dest, label=f"tessdata/{lang_code}")
        _print_ok(f"Idioma '{lang_code}' instalado correctamente.")
        return True
    except ConnectionError as exc:
        _print_error(f"No se pudo descargar '{lang_code}': {exc}")
        return False


def get_installed_languages() -> List[str]:
    """
    Lista los idiomas instalados en el Tesseract bundled.

    Returns:
        Lista de códigos de idioma disponibles.
    """
    if not TESSDATA_DIR.exists():
        return []

    return sorted(
        f.stem
        for f in TESSDATA_DIR.glob("*.traineddata")
    )


def cleanup() -> None:
    """
    Elimina la instalación local de Tesseract.

    Borra completamente el directorio vendor/tesseract/.
    """
    if VENDOR_DIR.exists():
        shutil.rmtree(VENDOR_DIR)
        _print_ok("Instalación local de Tesseract eliminada.")
    else:
        _print("No hay instalación local que eliminar.")


# ─────────────────────────────────────────────
# CLI del módulo
# ─────────────────────────────────────────────
def _cli() -> None:
    """
    Interfaz de línea de comandos para el gestor de Tesseract.

    Uso:
        python tesseract_manager.py [setup|status|add-lang|cleanup]
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="tesseract_manager",
        description="Gestor de Tesseract OCR integrado para ImageRD.",
    )

    sub = parser.add_subparsers(dest="command", help="Comando a ejecutar.")

    # setup
    setup_cmd = sub.add_parser("setup", help="Descargar e instalar Tesseract localmente.")
    setup_cmd.add_argument(
        "--langs", "-l",
        type=str,
        default="eng,spa,osd",
        help="Idiomas a descargar, separados por coma (default: eng,spa,osd).",
    )
    setup_cmd.add_argument(
        "--force", "-f",
        action="store_true",
        help="Forzar reinstalación.",
    )

    # status
    sub.add_parser("status", help="Mostrar estado de la instalación.")

    # add-lang
    add_cmd = sub.add_parser("add-lang", help="Agregar idioma adicional.")
    add_cmd.add_argument("lang", type=str, help="Código de idioma (e.g., fra, deu, por).")

    # cleanup
    sub.add_parser("cleanup", help="Eliminar instalación local de Tesseract.")

    args = parser.parse_args()

    if args.command == "setup":
        langs = [l.strip() for l in args.langs.split(",") if l.strip()]
        setup(languages=langs, force=args.force)

    elif args.command == "status":
        print(f"\n{'═' * 55}")
        print("  Tesseract Manager — Estado")
        print(f"{'═' * 55}\n")

        if is_bundled():
            _print_ok(f"Tesseract bundled: {get_tesseract_path()}")
            if _verify_installation():
                _print_ok("Estado: funcional ✓")
            else:
                _print_error("Estado: instalado pero no funcional ✗")
                missing = _diagnose_missing_libs()
                if missing:
                    _print_error("Librerías faltantes:")
                    for lib in missing:
                        _print_error(f"  → {lib}")
        else:
            _print("Tesseract bundled: no instalado")
            sys_tess = shutil.which("tesseract")
            if sys_tess:
                _print_ok(f"Tesseract del sistema: {sys_tess}")
            else:
                _print_warn("Tesseract del sistema: no encontrado")

        langs = get_installed_languages()
        if langs:
            _print_ok(f"Idiomas instalados: {', '.join(langs)}")
        else:
            _print("Idiomas instalados: ninguno")

        print()

    elif args.command == "add-lang":
        add_language(args.lang)

    elif args.command == "cleanup":
        cleanup()

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
