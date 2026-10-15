# Cubos espaciotemporales IDW

Un único generador para uno o hasta nueve contaminantes. Conserva la composición compacta, sin escalas visibles, con nombres centrados debajo, mapa frontal y líneas cartográficas extendidas por la profundidad.

## Ejecución

Desde la raíz del proyecto:

```powershell
uv run --group presentation python -m scripts.presentation.render_idw --help
uv run --group presentation python -m scripts.presentation.render_idw --preview-only
uv run --group presentation python -m scripts.presentation.render_idw
```

El resultado se guarda por defecto en `outputs/idw_nine`. La exportación crea `idw_cubes_scene_labels_full.mp4`, `idw_cubes_scene_labels_full_poster.png`, vistas del inicio/mitad/final y `idw_cubes_scene_labels_full_metadata.json`. Los vídeos ya existentes se conservan con sus nombres anteriores.

Para importar una caché cartográfica existente:

```powershell
uv run --group presentation python -m scripts.presentation.render_idw --map-cache outputs/idw_month --preview-only
```

Después la cartografía es autónoma en `outputs/idw_nine/cartography` y no se necesita `--map-cache`. Si no hay cartografía guardada, el generador consulta OpenStreetMap mediante Overpass y la almacena localmente. El mapa representa la cartografía disponible al descargarlo, no necesariamente la del año de las mediciones.

Para un solo contaminante o una selección:

```powershell
uv run --group presentation python -m scripts.presentation.render_idw --pollutant 8 --output outputs/presentation/no2
uv run --group presentation python -m scripts.presentation.render_idw --pollutant 8 --pollutant 14 --output outputs/presentation/no2_o3 --preview-only
```

Cambiar mes o selección requiere otra carpeta de salida: las cachés incompatibles se rechazan. Ejemplo:

```powershell
uv run --group presentation python -m scripts.presentation.render_idw --month 2024-02 --output outputs/presentation/febrero
```

Opciones: `--month` (YYYY-MM), `--output`, `--map-cache`, `--pollutant` (repetible), `--seconds` (62), `--fps` (24), `--config` (YAML visual) y `--preview-only`.

Selección predeterminada: NO₂ (8), NO (7), NOx (12), PM2.5 (9), PM10 (10), O₃ (14), CO (6), benceno (30), tolueno (20).

## Organización

- `render_idw.py`: único punto de entrada; importación de Click, decoradores y comando dentro de `if __name__ == '__main__':`.
- `data.py`: observaciones METRAQ, rejilla compartida, reconstrucción IDW y validación de cachés.
- `cartography.py`: mapa georreferenciado y caché cartográfica.
- `rendering.py`: escena 3D PyVista/VTK, texturas de las seis caras y ventana temporal.
- `configuration.py` y `style.yaml`: configuración visual validada, cámara, profundidad y composición.
- `common.py`: paleta, fuentes y codificación de vídeo.

Reutiliza los interpoladores y la agregación del proyecto. Solo admite registros observados y finitos, conserva ceros válidos y valores conocidos, y exige al menos dos celdas observadas en cada hora. La ventana contiene 24 registros horarios, incluido el instante actual; se cargan también las 23 horas anteriores al mes. No hay evaluación de sensores retenidos: es una presentación de reconstrucciones.

Cada contaminante conserva sus propios límites de color durante el mes, aunque las barras no se dibujan. El suavizado entre horas interpola colores para la animación; cada reconstrucción IDW se calcula independientemente por hora.

## Dependencias y verificación

Instala las dependencias específicas de presentación:

```powershell
uv sync --group presentation
```

Este grupo incluye PyVista/VTK y Pillow. La escena se renderiza fuera de pantalla y necesita un contexto gráfico OpenGL válido (comprobado en Windows en este equipo). FFmpeg debe estar en PATH para exportar MP4; las imágenes no lo necesitan. Fuentes: Segoe UI en Windows o DejaVu Sans como alternativa.

```powershell
uv run --group presentation pytest tests/test_presentation.py tests/test_presentation_3d.py
```

## Configuración visual

Los ajustes están en `style.yaml`; para otra configuración usa `--config ruta/al/estilo.yaml`.

- Cámara: `camera_azimuth`, `camera_elevation` (grados), `camera_view_angle`, `camera_zoom`.
- Cubos: `cube_depth`, `texture_size`, `map_blend`, `extrude_map_lines`.
- Composición: `width`, `height`, `body_width_fraction`, `header_fraction`, `footer_fraction`, `column_gap`, `row_gap`.
- Estilo: colores, tamaños de letra y `artifact_prefix` para distinguir las salidas.

NumPy prepara los campos, colores y ventanas temporales. Pillow prepara la cartografía y las texturas (redimensionado, mezcla y transparencia) y guarda las capturas PNG. PyVista/VTK renderiza la imagen completa: cubos, fondo, títulos, fechas y nombres, sin composición posterior ni rectángulos para las etiquetas. FFmpeg codifica el vídeo.

Todos los cubos se colocan en una única escena 3D, con una sola cámara de perspectiva y un búfer de profundidad compartido. La biblioteca ajusta el encuadre al conjunto y resuelve visibilidad y coordenadas de textura. Las etiquetas son actores de texto 3D, centrados bajo el borde frontal y dentro de la misma escena: comparten perspectiva y oclusión con los cubos. `label_height` controla su altura en unidades de la escena y `label_color` su color; `label_font_size` controla la resolución de su texto rasterizado. La separación entre columnas y filas se controla con `column_gap` y `row_gap`, en unidades de la escena (la altura de cada cubo es 1). Se usan superficies texturizadas, sin modificar las reconstrucciones IDW ni sus escalas.

Las pruebas de orientación que necesitan OpenGL se activan con `METRAQ_TEST_3D=1`. La configuración completa usada en cada exportación queda en sus metadatos.

## Reutilizar datos y mapa en una salida nueva

```powershell
uv run --group presentation python -m scripts.presentation.render_idw --data-cache outputs/idw_nine --output outputs/presentation/nueva_version
```

`--data-cache` lee las reconstrucciones existentes sin copiarlas ni recalcularlas y reutiliza su cartografía; `--map-cache` permite elegir otra caché cartográfica compatible. El mes, los contaminantes y su orden deben coincidir con los de la caché (indícalos con `--month` y `--pollutant` cuando no sean los predeterminados). Los vídeos y metadatos nuevos se guardan en `--output`; usa una carpeta nueva para conservar la versión anterior. Para comprobar solo imágenes, añade `--preview-only`.
