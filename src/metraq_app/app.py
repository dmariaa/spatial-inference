"""Run with: uv run --group app streamlit run src/metraq_app/app.py"""
from datetime import datetime, time, timedelta
import sys
import os
from pathlib import Path

# Also works when Streamlit executes this file directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from metraq_app.data_source import POLLUTANTS, load_catalog, load_sensors, load_measurements
from metraq_app.grid import make_grid, aggregate_cells
from metraq_app.map import build_map
from streamlit.components.v1 import declare_component

native_map = declare_component("metraq_svg_map", path=str(Path(__file__).parent / "svg_view"))

catalog_cached = st.cache_data(ttl=3600)(load_catalog)
sensors_cached = st.cache_data(ttl=3600)(load_sensors)
measurements_cached = st.cache_data(ttl=300)(load_measurements)
grid_cached = st.cache_data(show_spinner=False)(make_grid)

st.set_page_config(page_title="METRAQ · Madrid", page_icon="🌍", layout="wide")
st.markdown("""
<style>
    .stMainBlockContainer { padding-top: 3rem; padding-bottom: 0.5rem; }
    [data-testid="stVerticalBlock"] { gap: 0.5rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
    [data-testid="stColumn"] { min-width: 0 !important; width: 0 !important; flex: 1 1 0 !important; }
    [data-testid="stHorizontalBlock"]:has(iframe) > [data-testid="stColumn"]:first-child { flex: 3 1 0 !important; }
    [data-testid="stHorizontalBlock"]:has(iframe) > [data-testid="stColumn"]:last-child { flex: 2 1 0 !important; }
    .metraq-native-map { width: 100%; }
    .metraq-native-map svg { display: block; width: 100%; height: auto; aspect-ratio: 1; background: #fff; }
    .metraq-cell:hover, .metraq-cell:focus { stroke: #111; stroke-width: 2; outline: none; }
    .metraq-scale { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; font-size: 0.75rem; }
    .metraq-scale > div { flex: 1; height: 10px; border-radius: 3px; }
    .metraq-attribution { font-size: 0.65rem; text-align: right; margin-top: 0.25rem; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)
st.markdown("### Calidad del aire · Madrid")
st.sidebar.caption("Horas según las marcas temporales del origen.")
source = st.sidebar.selectbox("Origen de datos", ["db", "files"], key="source",
    index=1 if os.getenv("METRAQ_APP_SOURCE") == "files" else 0,
    format_func=lambda value: {"db": "Base de datos", "files": "Dataset METRAQ (CSV)"}[value])
try:
    with st.spinner("Leyendo catálogo de datos (la primera carga puede tardar)…"):
        catalog = catalog_cached(source)
except Exception:
    st.error("No se pudo leer el origen. Comprueba la conexión a la BD o selecciona Dataset METRAQ (CSV).")
    st.info("Los CSV deben estar en data/METRAQ o en la carpeta indicada por METRAQ_DATA_DIR.")
    st.stop()
if catalog.empty:
    st.warning("No hay contaminantes disponibles en este origen.")
    st.stop()
ids = sorted(catalog.magnitude_id.astype(int).tolist())
magnitude = st.sidebar.selectbox("Contaminante", ids, index=ids.index(12) if 12 in ids else 0,
    format_func=lambda mid: POLLUTANTS[mid][0])
metadata = catalog.set_index("magnitude_id").loc[magnitude]
first, last = metadata.first_date.to_pydatetime(), metadata.last_date.to_pydatetime()
selection_key = f"selected-{source}-{magnitude}"
if selection_key not in st.session_state:
    st.session_state[selection_key] = first

def move_hour(delta):
    st.session_state[selection_key] = min(last, max(first, st.session_state[selection_key] + timedelta(hours=delta)))
    st.session_state[f"date-{selection_key}"] = st.session_state[selection_key].date()
    st.session_state[f"hour-{selection_key}"] = st.session_state[selection_key].hour

selected = st.session_state[selection_key]
st.session_state.setdefault(f"date-{selection_key}", selected.date())
st.session_state.setdefault(f"hour-{selection_key}", selected.hour)
day = st.sidebar.date_input("Fecha", value=None, min_value=first.date(),
                            max_value=last.date(), key=f"date-{selection_key}")
hour = st.sidebar.selectbox("Hora", list(range(24)), index=None,
    format_func=lambda value: f"{value:02d}:00", key=f"hour-{selection_key}")
if day is None or hour is None:
    st.info("Selecciona una fecha y una hora para ver las medidas.")
    st.stop()
timestamp = datetime.combine(day, time(hour))
st.session_state[selection_key] = timestamp
previous, following = st.sidebar.columns(2)
previous.button("← 1 h", on_click=move_hour, args=(-1,), disabled=timestamp <= first)
following.button("1 h →", on_click=move_hour, args=(1,), disabled=timestamp >= last)
st.sidebar.caption(f"Disponibilidad: {first:%d/%m/%Y %H:%M} — {last:%d/%m/%Y %H:%M}. Puede haber huecos.")
cell_size = st.sidebar.selectbox("Tamaño de celda", [1000, 500, 2000], format_func=lambda value: f"{value} m")
auto_scale = st.sidebar.checkbox("Ajustar colores a esta hora", value=False)
show_sensors = st.sidebar.checkbox("Mostrar sensores con datos", value=True)
if st.sidebar.button("Actualizar datos"):
    st.cache_data.clear()
    st.rerun()
try:
    with st.spinner("Cargando observaciones…"):
        sensors = sensors_cached(source, tuple(ids))
        ctx = grid_cached(sensors, cell_size)
        measurements = measurements_cached(source, magnitude, timestamp)
        cells, observations = aggregate_cells(ctx, sensors, measurements)
except Exception:
    st.error("No se pudieron cargar las observaciones o la geometría. Comprueba el origen y la fecha seleccionada.")
    st.stop()
name, unit = POLLUTANTS[magnitude]
st.subheader(f"{name} · {timestamp:%d/%m/%Y %H:%M}")
left, middle, right = st.columns(3)
left.metric("Sensores con datos", len(observations))
middle.metric("Celdas con datos", len(cells))
right.metric("Media entre sensores", f"{observations.value.mean():.2f} {unit}" if len(observations) else "—")
if cells.empty:
    st.info("No hay medidas válidas para este contaminante en esta hora.")
bounds = None if auto_scale else (float(metadata.low), float(metadata.high))
if bounds is not None and bounds[0] == bounds[1]:
    bounds = (bounds[0], bounds[0] + 1)
map_panel, station_panel = st.columns([3, 2], gap="medium")
with map_panel:
    native_map(markup=build_map(ctx, cells, observations, unit, bounds, show_sensors), key="native-map")
    st.caption("Color: media por celda. Sin relleno: sin medidas. "
               + ("Escala automática para esta hora." if auto_scale else "Escala fija por contaminante."))
with station_panel:
    st.markdown("**Medidas por estación**")
    table = observations[["sensor_id", "name", "value"]].rename(
        columns={"sensor_id": "ID", "name": "Estación", "value": f"Valor ({unit})"})
    st.dataframe(table, hide_index=True, width="stretch", height=390,
                 column_order=["Estación", f"Valor ({unit})", "ID"],
                 column_config={f"Valor ({unit})": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Descargar CSV", table.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"metraq_{magnitude}_{timestamp:%Y%m%dT%H%M}.csv", mime="text/csv")

