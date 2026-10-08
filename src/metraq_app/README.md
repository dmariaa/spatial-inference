# METRAQ App

Visor local de observaciones horarias de Madrid: backend Python/FastAPI y frontend
HTML, CSS y JavaScript, sin Streamlit ni Plotly en la aplicación.

```powershell
uv run --group app python -m metraq_app --source files --port 8501
```

Abre http://127.0.0.1:8501. Para empezar desde la BD, usa `--source db`.
El origen también se puede cambiar en la interfaz. Los CSV se buscan en
`data/METRAQ` o en `METRAQ_DATA_DIR`. Se reutiliza la conexión existente para la BD.
La primera carga de archivos recorre las columnas del catálogo por lotes.
El catálogo, la geometría y hasta 128 consultas horarias quedan en caché;
«Actualizar datos» limpia esa caché. El mapa base necesita acceso a Internet.

## Archivos

- `app.py`: API y servicio de los archivos estáticos.
- `__main__.py`: comando Click para iniciar Uvicorn, limitado a localhost por defecto.
- `frontend/index.html`: estructura de la página.
- `frontend/styles.css`: todos los estilos de la interfaz, del mapa y de la escala.
- `frontend/app.js`: controles, consultas, cancelación de respuestas antiguas y CSV.
- `map.py`: SVG del grid y teselas OpenStreetMap georreferenciadas, sin CSS/JS incrustados.
- `data_source.py`, `grid.py`: acceso a datos y agregación de observaciones.

El grid es fijo dentro de un origen, con EPSG:25830, celdas de 1000 m y márgenes
3000/2000 m por defecto. Se muestra completo en un visor cuadrado; las celdas no se
deforman. Si coinciden varios sensores, el color representa su media después de
promediar duplicados por sensor. Cero es una medida válida; ausencia es sin relleno.
La tabla y el CSV incluyen estación, valor e ID. En pantallas estrechas el ID de
la tabla se oculta para dar espacio al valor; se conserva en el CSV.

Las fechas usan las marcas temporales del origen sin conversión de zona horaria.
El calendario se limita al intervalo de cada contaminante, con posibles huecos.
Por defecto los colores se ajustan a la hora seleccionada; la casilla «Escala fija»
activa el rango del origen. La escala fija usa el rango del origen; puede diferir entre BD y CSV. La automática
usa los valores de la hora. La vista «Solo observaciones» no calcula estimaciones; los métodos opcionales se describen abajo.

La metadata de los archivos reutiliza métodos internos de MetraqFiles,
concentrados en `data_source.py` hasta disponer de una API pública de catálogo.

Pruebas: `uv run --group app --group dev pytest tests/test_metraq_app.py tests/test_metraq_app_interpolation.py`.
El prototipo previo con Streamlit quedó guardado en el commit `5f7c699`.


## Interpolación de la hora seleccionada

El selector permite Solo observaciones, IDW, KRG y DIP-CNN. Todos reciben las
mismas medias de las celdas observadas en esa hora, con una máscara que conserva
los ceros válidos. IDW y KRG reutilizan los interpoladores del proyecto. Las
medidas se mantienen exactamente en sus celdas; solo se estiman las demás.
El cursor distingue media observada y estimación del método. El modo de colores
automáticos utiliza el rango de toda la superficie mostrada.

DIP-CNN ejecuta el DipOptimizer existente con UNet3D (dos niveles, 16 canales
base, kernel 1×3×3), ruido aleatorio y ajuste MAE sobre las observaciones de una
hora. Normaliza con media/desviación de esas observaciones y devuelve unidades
originales. Se puede elegir 100, 250 o 500 iteraciones. Utiliza la última
superficie, sin selección por validación ni ensemble. Se ejecuta localmente en
CPU, con una sola optimización a la vez, y no fija semillas.

Este visor usa todas las medidas disponibles y no constituye una evaluación
TRAIN/VALIDATION/TEST ni reproduce las campañas de 24 horas. La tabla de estaciones
y su CSV continúan mostrando medidas, no predicciones. Con una sola celda observada
o valores constantes se muestra un campo constante y se indica en la interfaz.
Sin medidas no se inventa una superficie. No se recortan estimaciones a los
límites de la escala fija; los valores reales se pueden consultar en el cursor.

Hasta 32 superficies quedan en caché por origen, contaminante, hora, grid, método
e iteraciones. Cambiar escala o visibilidad de sensores no vuelve a optimizar DIP.
«Actualizar datos» limpia las superficies junto con los datos; una reconstrucción
DIP posterior puede cambiar porque el ruido y la inicialización son aleatorios.
