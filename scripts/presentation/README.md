# Cubos espaciotemporales IDW

Un único generador para uno o hasta nueve contaminantes. Conserva la composición compacta, sin escalas visibles, con nombres centrados debajo, mapa frontal y líneas cartográficas extendidas por la profundidad.

## Ejecución

Desde la raíz del proyecto:

```powershell
uv run python -m scripts.presentation.render_idw --help
uv run python -m scripts.presentation.render_idw --preview-only
uv run python -m scripts.presentation.render_idw
```

El resultado se guarda por defecto en `outputs/idw_nine`. La exportación crea `idw_cubes.mp4`, `poster.png`, vistas del inicio/mitad/final y metadatos. Los vídeos ya existentes se conservan con sus nombres anteriores.

Para importar una caché cartográfica existente:

```powershell
uv run python -m scripts.presentation.render_idw --map-cache outputs/idw_month --preview-only
```

Después la cartografía es autónoma en `outputs/idw_nine/cartography` y no se necesita `--map-cache`. Si no hay cartografía guardada, el generador consulta OpenStreetMap mediante Overpass y la almacena localmente. El mapa representa la cartografía disponible al descargarlo, no necesariamente la del año de las mediciones.

Para un solo contaminante o una selección:

```powershell
uv run python -m scripts.presentation.render_idw --pollutant 8 --output outputs/presentation/no2
uv run python -m scripts.presentation.render_idw --pollutant 8 --pollutant 14 --output outputs/presentation/no2_o3 --preview-only
```

Cambiar mes o selección requiere otra carpeta de salida: las cachés incompatibles se rechazan. Ejemplo:

```powershell
uv run python -m scripts.presentation.render_idw --month 2024-02 --output outputs/presentation/febrero
```

Opciones: `--month` (YYYY-MM), `--output`, `--map-cache`, `--pollutant` (repetible), `--seconds` (62), `--fps` (24) y `--preview-only`.

Selección predeterminada: NO₂ (8), NO (7), NOx (12), PM2.5 (9), PM10 (10), O₃ (14), CO (6), benceno (30), tolueno (20).

## Organización

- `render_idw.py`: único punto de entrada; importación de Click, decoradores y comando dentro de `if __name__ == '__main__':`.
- `data.py`: observaciones METRAQ, rejilla compartida, reconstrucción IDW y validación de cachés.
- `cartography.py`: mapa georreferenciado y caché cartográfica.
- `rendering.py`: cámara, composición, ventana temporal, imágenes y vídeo.
- `common.py`: paleta, fuentes, proyección de texturas y codificación de vídeo.

Reutiliza los interpoladores y la agregación del proyecto. Solo admite registros observados y finitos, conserva ceros válidos y valores conocidos, y exige al menos dos celdas observadas en cada hora. La ventana contiene 24 registros horarios, incluido el instante actual; se cargan también las 23 horas anteriores al mes. No hay evaluación de sensores retenidos: es una presentación de reconstrucciones.

Cada contaminante conserva sus propios límites de color durante el mes, aunque las barras no se dibujan. El suavizado entre horas interpola colores para la animación; cada reconstrucción IDW se calcula independientemente por hora.

## Dependencias y verificación

Usa las dependencias del proyecto, incluyendo Pillow (ya instalado a través de torchvision). FFmpeg debe estar en PATH para exportar MP4; las imágenes no lo necesitan. Fuentes: Segoe UI en Windows o DejaVu Sans como alternativa.

```powershell
uv run pytest tests/test_presentation.py
```
