'use strict';
const byId = id => document.getElementById(id);
const ui = Object.fromEntries(['source', 'pollutant', 'method', 'dip-epochs', 'dip-options', 'map-note', 'date', 'hour', 'cell-size', 'fixed-scale', 'show-sensors', 'previous', 'next', 'refresh', 'coverage', 'status', 'results', 'selection', 'sensor-count', 'cell-count', 'mean', 'map', 'stations', 'value-heading', 'download'].map(id => [id, byId(id)]));
const state = {catalog: [], view: null, controller: null, generation: 0};
for (let hour = 0; hour < 24; hour++) {
    ui.hour.add(new Option(`${String(hour).padStart(2, '0')}:00`, String(hour)));
}

function status(message, error = false) {
    ui.status.textContent = message;
    ui.status.classList.toggle('error', error);
    ui.status.hidden = !message;
}

function invalidate() {
    state.generation++;
    state.controller?.abort();
    state.controller = null;
    state.view = null;
    clearCellSelection();
    ui.results.hidden = true;
    ui.download.disabled = true;
}

async function request(url, signal, options = {}) {
    const response = await fetch(url, {...options, signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'No se pudo completar la consulta.');
    return data;
}

function currentPollutant() {
    return state.catalog.find(item => item.id === Number(ui.pollutant.value));
}

function timestamp() {
    return `${ui.date.value}T${String(ui.hour.value).padStart(2, '0')}:00:00`;
}

function displayDate(value) {
    const [date, time] = value.split('T');
    return `${date.split('-').reverse().join('/')} ${time.slice(0, 5)}`;
}

function updateNavigation() {
    const item = currentPollutant();
    const value = timestamp();
    ui.previous.disabled = !item || value <= item.first;
    ui.next.disabled = !item || value >= item.last;
}

function selectPollutant() {
    const item = currentPollutant();
    if (!item) return;
    ui.date.min = item.first.slice(0, 10);
    ui.date.max = item.last.slice(0, 10);
    if (!ui.date.value || timestamp() < item.first || timestamp() > item.last) {
        ui.date.value = item.first.slice(0, 10);
        ui.hour.value = String(Number(item.first.slice(11, 13)));
    }
    ui.coverage.textContent = `Disponible: ${displayDate(item.first)} — ${displayDate(item.last)}. Horas del origen; puede haber huecos.`;
    updateNavigation();
}

function catalogControls(disabled) {
    for (const id of ['pollutant', 'method', 'dip-epochs', 'date', 'hour', 'cell-size', 'fixed-scale', 'show-sensors', 'previous', 'next']) ui[id].disabled = disabled;
}

async function loadCatalog() {
    invalidate();
    catalogControls(true);
    state.catalog = [];
    ui.pollutant.replaceChildren();
    status('Leyendo catálogo… La primera carga puede tardar.');
    const generation = state.generation;
    const controller = new AbortController();
    state.controller = controller;
    const timeout = setTimeout(() => controller.abort('timeout'), 60000);
    try {
        const data = await request(`/api/catalog?source=${encodeURIComponent(ui.source.value)}`, controller.signal);
        if (generation !== state.generation) return;
        state.catalog = data.pollutants;
        if (!state.catalog.length) throw new Error('Este origen no contiene contaminantes disponibles.');
        for (const item of state.catalog) ui.pollutant.add(new Option(item.name, item.id));
        ui.pollutant.value = String(state.catalog.some(item => item.id === 12) ? 12 : state.catalog[0].id);
        catalogControls(false);
        selectPollutant();
        await loadView();
    } catch (error) {
        if (generation !== state.generation) return;
        status(controller.signal.aborted ? 'La consulta ha tardado demasiado. Puedes cambiar de origen o actualizar.' : error.message, true);
    } finally {
        clearTimeout(timeout);
    }
}

async function loadView() {
    invalidate();
    const item = currentPollutant();
    if (!item) return;
    const generation = state.generation;
    const selectedTime = timestamp();
    updateNavigation();
    if (!ui.date.value || selectedTime < item.first || selectedTime > item.last) {
        status('Selecciona una fecha y hora dentro del intervalo disponible.', true);
        return;
    }
    const controller = new AbortController();
    state.controller = controller;
    const timeout = setTimeout(() => controller.abort('timeout'), ui.method.value === 'dip-cnn' ? 300000 : 60000);
    status(ui.method.value === 'dip-cnn' ? 'Calculando DIP-CNN para esta hora… La primera reconstrucción puede tardar.' : 'Cargando observaciones e interpolación…');
    const params = new URLSearchParams({
        source: ui.source.value,
        magnitude: item.id,
        timestamp: selectedTime,
        cell_size: ui['cell-size'].value,
        auto_scale: !ui['fixed-scale'].checked,
        show_sensors: ui['show-sensors'].checked,
        method: ui.method.value,
        dip_epochs: ui['dip-epochs'].value
    });
    try {
        const data = await request(`/api/view?${params}`, controller.signal);
        if (generation !== state.generation) return;
        state.view = data;
        ui.selection.textContent = `${data.name} · ${displayDate(data.timestamp)} · ${data.method}`;
        ui['sensor-count'].textContent = data.stations.length;
        ui['cell-count'].textContent = data.cell_count;
        ui.mean.textContent = data.mean === null ? '—' : `${data.mean.toFixed(2)} ${data.unit}`;
        ui.map.innerHTML = data.map;
        for (const cell of ui.map.querySelectorAll('.metraq-cell')) {
            cell.dataset.details = cell.querySelector('title')?.textContent || 'Sin datos';
            cell.setAttribute('aria-label', cell.dataset.details);
            cell.setAttribute('role', 'button');
        }
        for (const title of ui.map.querySelectorAll('svg title')) title.remove();
        ui['map-note'].textContent = data.interpolated_count
            ? `${data.cell_count} celdas observadas · ${data.interpolated_count} estimadas (${data.method}). Pasa el cursor para distinguir medidas y estimaciones.`
            : 'Color: media por celda. Sin relleno: sin medidas. Pasa el cursor para ver los valores.';
        if (data.constant_field) {
            ui['map-note'].textContent += ' Medidas iguales o una sola celda observada: campo constante.';
        } else if (data.method === 'DIP-CNN' && data.interpolated_count) {
            ui['map-note'].textContent += ` Una hora, ${data.dip_epochs} iteraciones; última superficie, sin selección por validación.`;
        }
        ui['value-heading'].textContent = `Valor (${data.unit})`;
        ui.stations.replaceChildren();
        for (const station of data.stations) {
            const row = document.createElement('tr');
            row.dataset.cell = station.cell_id;
            row.tabIndex = 0;
            row.setAttribute('aria-label', `Seleccionar estación ${station.name}`);
            for (const value of [station.name, station.value.toFixed(2), station.id]) {
                const cell = document.createElement('td');
                cell.textContent = value;
                row.append(cell);
            }
            ui.stations.append(row);
        }
        ui.results.hidden = false;
        requestAnimationFrame(layoutWorkspace);
        ui.download.disabled = !data.stations.length;
        status(data.stations.length ? '' : 'No hay medidas válidas para esta hora.');
    } catch (error) {
        if (generation !== state.generation) return;
        status(controller.signal.aborted ? 'La consulta ha tardado demasiado. Puedes cambiar de origen o actualizar.' : error.message, true);
    } finally {
        clearTimeout(timeout);
    }
}

function moveHour(delta) {
    const date = new Date(`${timestamp()}Z`);
    date.setUTCHours(date.getUTCHours() + delta);
    const item = currentPollutant();
    const value = date.toISOString().slice(0, 19);
    if (value < item.first || value > item.last) return;
    ui.date.value = value.slice(0, 10);
    ui.hour.value = String(Number(value.slice(11, 13)));
    loadView();
}

function clearCellSelection() {
    ui.map.querySelector('.metraq-cell.is-selected')?.classList.remove('is-selected');
    ui.map.querySelector('.metraq-cell.is-hovered')?.classList.remove('is-hovered');
    ui.map.querySelector('.cell-details')?.remove();
    for (const row of ui.stations.querySelectorAll('.is-selected')) row.classList.remove('is-selected');
}

function showCell(cell, pinned = false) {
    clearCellSelection();
    cell.classList.add(pinned ? 'is-selected' : 'is-hovered');
    if (pinned) {
        const rows = [...ui.stations.rows].filter(row => row.dataset.cell === cell.dataset.cell);
        for (const row of rows) row.classList.add('is-selected');
        if (rows.length) {
            const container = ui.stations.closest('.table-scroll');
            const rowBounds = rows[0].getBoundingClientRect();
            const bounds = container.getBoundingClientRect();
            if (rowBounds.top < bounds.top || rowBounds.bottom > bounds.bottom) {
                container.scrollTop += rowBounds.top - bounds.top - container.querySelector('thead').getBoundingClientRect().height;
            }
        }
    }
    const details = document.createElement('div');
    details.className = pinned ? 'cell-details is-pinned' : 'cell-details';
    details.setAttribute('role', 'status');
    const text = document.createElement('div');
    text.textContent = cell.dataset.details;
    details.append(text);
    if (pinned) {
        const close = document.createElement('button');
        close.type = 'button';
        close.textContent = '×';
        close.setAttribute('aria-label', 'Cerrar información de celda');
        close.addEventListener('click', clearCellSelection);
        details.append(close);
    }
    ui.map.append(details);
}

function selectCell(cell) {
    if (cell.classList.contains('is-selected')) clearCellSelection();
    else showCell(cell, true);
}

function cellAtPointer(event) {
    if (event.target.closest('.cell-details')) return null;
    return event.target.closest('.metraq-cell') || document.elementsFromPoint(event.clientX, event.clientY)
        .find(element => element.classList.contains('metraq-cell'));
}

ui.map.addEventListener('pointermove', event => {
    if (event.pointerType === 'touch' || ui.map.querySelector('.is-selected')) return;
    const cell = cellAtPointer(event);
    if (!cell) clearCellSelection();
    else if (!cell.classList.contains('is-hovered')) showCell(cell);
});
ui.map.addEventListener('pointerleave', () => {
    if (!ui.map.querySelector('.is-selected')) clearCellSelection();
});
ui.map.addEventListener('focusin', event => {
    const cell = event.target.closest('.metraq-cell');
    if (cell && !ui.map.querySelector('.is-selected')) showCell(cell);
});
ui.map.addEventListener('focusout', () => {
    if (!ui.map.querySelector('.is-selected')) clearCellSelection();
});
ui.map.addEventListener('click', event => {
    const cell = cellAtPointer(event);
    if (cell) selectCell(cell);
});
ui.map.addEventListener('keydown', event => {
    if (event.key === 'Escape') clearCellSelection();
    const cell = event.target.closest('.metraq-cell');
    if (cell && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        selectCell(cell);
    }
});

function selectStationRow(row) {
    if (!row) return;
    const cell = ui.map.querySelector(`.metraq-cell[data-cell="${row.dataset.cell}"]`);
    if (cell) selectCell(cell);
}

ui.stations.addEventListener('click', event => selectStationRow(event.target.closest('tr')));
ui.stations.addEventListener('keydown', event => {
    if (event.key === 'Escape') clearCellSelection();
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectStationRow(event.target.closest('tr'));
    }
});

ui.source.addEventListener('change', loadCatalog);
ui.pollutant.addEventListener('change', () => {
    selectPollutant();
    loadView();
});
ui.method.addEventListener('change', () => {
    ui['dip-options'].hidden = ui.method.value !== 'dip-cnn';
    loadView();
});
for (const id of ['date', 'hour', 'cell-size', 'fixed-scale', 'show-sensors', 'dip-epochs']) ui[id].addEventListener('change', loadView);
ui.previous.addEventListener('click', () => moveHour(-1));
ui.next.addEventListener('click', () => moveHour(1));
byId('controls').addEventListener('submit', event => {
    event.preventDefault();
    loadView();
});
ui.refresh.addEventListener('click', async () => {
    invalidate();
    try {
        await request('/api/refresh', undefined, {method: 'POST'});
        await loadCatalog();
    } catch (error) {
        status(error.message, true);
    }
});

function csvCell(value) {
    let text = String(value);
    if (/^[=+@-]/.test(text) && typeof value === 'string') text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
}

ui.download.addEventListener('click', () => {
    if (!state.view) return;
    const data = state.view;
    const rows = [['Estación', `Valor (${data.unit})`, 'ID'], ...data.stations.map(item => [item.name, item.value, item.id])];
    const blob = new Blob(['\uFEFF' + rows.map(row => row.map(csvCell).join(',')).join('\r\n')], {type: 'text/csv;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `metraq_${ui.pollutant.value}_${data.timestamp.replaceAll(/[-:]/g, '')}.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
(async () => {
    try {
        const config = await request('/api/config');
        ui.source.value = config.default_source;
        await loadCatalog();
    } catch (error) {
        status(error.message, true);
    }
})();


// Size the square map and the independently scrolling table to the actual
// remaining viewport, including when the controls wrap in a narrow window.
function layoutWorkspace() {
    if (ui.results.hidden) return;
    const workspace = document.querySelector('.workspace');
    const available = Math.max(180, window.innerHeight - workspace.getBoundingClientRect().top - 16);
    const mapPanel = document.querySelector('.map-panel');
    const footer = ['.metraq-scale', '.metraq-attribution', '.map-note']
        .reduce((sum, selector) => sum + document.querySelector(selector).getBoundingClientRect().height, 24);
    const side = Math.floor(Math.min(mapPanel.clientWidth, Math.max(140, available - footer)));
    const headingHeight = document.querySelector('.panel-heading').getBoundingClientRect().height;
    document.documentElement.style.setProperty('--map-side', `${side}px`);
    document.documentElement.style.setProperty('--table-height', `${Math.floor(available - headingHeight - 2)}px`);
}

window.addEventListener('resize', layoutWorkspace);
new ResizeObserver(() => requestAnimationFrame(layoutWorkspace)).observe(document.querySelector('.controls'));
