# METRAQ App

Prototipo local para explorar observaciones horarias sobre el grid de Madrid.

```powershell
uv run --group app streamlit run src/metraq_app/app.py
```

Selecciona base de datos (configuración existente de METRAQ) o archivos CSV.
Los archivos se buscan en `data/METRAQ`; se puede cambiar la carpeta con
`METRAQ_DATA_DIR`. La primera carga del catálogo de archivos recorre las columnas
necesarias por lotes y puede tardar; los resultados quedan en caché una hora.
El mapa base necesita acceso a Internet para cargar las teselas.

El grid permanece fijo para todas las fechas y contaminantes de un mismo origen,
con EPSG:25830, celdas de 1000 m y márgenes de 3000/2000 m por defecto.
Si varios sensores ocupan la misma celda, se muestra su media, después de promediar
posibles duplicados por sensor. Un cero observado es un valor válido; la ausencia
de datos queda sin relleno. No se generan predicciones ni interpolaciones en la app;
se visualizan los valores que proporciona el backend.

Las fechas usan las marcas temporales del origen sin conversión de zona horaria.
El catálogo limita el calendario al intervalo disponible por contaminante; puede
haber huecos. Los colores fijos usan el rango del origen, por lo que los colores
pueden diferir entre BD y CSV. El modo automático facilita ver contrastes de una hora.

La lectura de metadatos CSV reutiliza métodos internos de MetraqFiles, concentrados
en data_source.py; una futura API pública de catálogo permitiría sustituirlos ahí.

El mapa se dibuja como SVG adaptable al ancho del panel, con teselas OpenStreetMap georreferenciadas. El grid completo aparece sin cámara ni PAN; al pasar el cursor por una celda se muestran sus medidas. La escala de color y la atribución quedan fuera del grid para no taparlo.
