# Kernel convexo con viento direccional

El viento es opcional y exclusivo del kernel convexo. Se reutilizan los datos
PyG, el entrenador y el evaluador comunes. La GNN no recibe canales nuevos.
Los checkpoints anteriores se cargan sin conversiones y el modo por defecto
sigue siendo `--wind-mode none`.

## Modelo

Para cada fuente visible i y receptor j:

```text
s_ij = -p log((d_ij + epsilon) / d0) + beta a_ij [+ correccion_radial(d_ij)]
a_ij = ((r_j - r_i) · w_i) / ((d_ij + epsilon) (||w_i|| + u0))
alpha_ij = softmax_i(s_ij)
prediccion_j = sum_i(alpha_ij * observacion_i)
```

- Las posiciones usan metros y ejes este/norte; la posición PyG es (col, -fila).
- u/v conservan sus unidades físicas m/s. No se centran ni estandarizan: hacerlo
  alteraría la dirección del viento. `u0=1 m/s` suaviza el término cerca de la calma.
- beta es no negativo, aprendido e inicializado en 0.01 (casi isotrópico).
- La calma válida y el viento ausente producen un término direccional cero.
  La disponibilidad permite distinguir ambos casos en los datos.
- Las observaciones AQ visibles se copian exactamente; los demás resultados
  siguen siendo combinaciones convexas de las fuentes visibles.
- Se cachean los desplazamientos y se calcula todo el lote con tensores.

La primera ablación propuesta es `--mode power --wind-mode directional`: añade
solo beta al exponente p aprendido. La corrección neuronal radial sigue
disponible con `--mode neural`. `fixed` fija solo p=2: con viento activo beta
sí se entrena; sin viento conserva su comportamiento de evaluación sin ajuste.

## Caché meteorológica

`scripts/build_gnn_wind_cache.py` consume las exportaciones mensuales de
`scripts/prepare_meteorology_dataset.py`, junto con su `scope.json`. Reutiliza
811/812, derivados previamente con `u=-VV*sin(DV), v=-VV*cos(DV)`; no vuelve
a convertir velocidad y dirección. Valida m/s y grados en el catálogo de origen.

Se necesita un catálogo CSV/parquet con `id, utm_x, utm_y`, en EPSG:25830,
que incluya las estaciones AQ de la caché y las meteorológicas del periodo.
La preparación usa solo archivos locales y reconstruye/valida la misma rejilla.
La opción por defecto `--mapping station` asigna los u/v existentes directamente
a cada celda AQ, sin interpolación espacial adicional. Si varias estaciones AQ
comparten celda se promedian sus vectores completos. `--mapping idw` queda como
alternativa explícita para estaciones fuente ubicadas en otros puntos.
En este piloto los valores interpolados de origen proceden de Ordinary Kriging,
según lo indicado por el usuario; `--source-interpolation` registra esa procedencia.
No se utilizan valores AQ ni particiones para preparar el viento.

```powershell
uv run python scripts/build_gnn_wind_cache.py --cache cache/metraq/no2-2010-2024 --meteorology-dir output/meteorology_admitted_2019_2024 --sensors <catalogo.csv> --start "2015-01-01" --end "2024-12-31 23:00" --mapping station --source-interpolation "Ordinary Kriging (user-provided)" --allow-missing-months --output cache/metraq/wind-kriging-direct-2015-2024
```

La configuración del piloto acepta explícitamente meses ausentes: el exportado
local cubre 2019–2024. En 2015–2018 se conserva el viento como ausente y el término
direccional es cero. No se eliminan ventanas AQ ni se rellenan huecos del dataset.
La caché conserva los u/v convertidos previamente desde velocidad/dirección,
incluidos sus valores observados y los estimados mediante Ordinary Kriging.

`--observed-only` admite únicamente pares cuyos componentes proceden de
observaciones originales. El valor por defecto también acepta pares previamente
interpolados y registra esa política. Las horas sin pares completos quedan
ausentes; un vector cero válido permanece disponible. Se guardan el número de
estaciones usadas y originales por hora, los archivos de origen y la cobertura.

La caché contiene únicamente (hora, celda AQ, componente), con huellas de
contenido. El evaluador exige exactamente la identidad utilizada al entrenar,
aunque la carpeta se traslade a otro equipo. No es necesario conservar en ese
equipo las rutas originales de los parquets para leer la caché.

## Entrenamiento y evaluación

Usar el mismo resumen GNN de partición que el piloto anterior:

```powershell
uv run python scripts/train/train_convex_kernel.py --cache cache/metraq/no2-2010-2024 --output output/convex-kernel/pilot-power-wind --mode power --wind-mode directional --wind-cache cache/metraq/wind-kriging-direct-2015-2024 --partition-summary <resumen-gnn.json> --seed 312537310 --batch-size 64 --epochs 100 --device cuda
```

Se mantienen TRAIN 2015–2022, validación 2023, las ventanas AQ y la normalización
ajustada solo con TRAIN. Se guarda beta, p, cobertura de viento y configuración.
La selección sigue usando MAE de validación y la paciencia del entrenador común.
W&B solo se utiliza si se solicita con `--wandb`; para un piloto sin publicación,
añadir `--wandb --wandb-mode offline`.

El siguiente comando es para evaluar posteriormente el checkpoint ya seleccionado;
el entrenamiento no lo invoca:

```powershell
uv run python scripts/evaluation/evaluate_cached_model.py --run-dir output/convex-kernel/pilot-power-wind --cache cache/metraq/no2-2010-2024 --wind-cache cache/metraq/wind-kriging-direct-2015-2024 --reference-summary <resumen-gnn.json> --output output/convex-kernel/pilot-power-wind-test --device cuda
```

El evaluador conserva las comprobaciones de partición, normalización, checkpoint,
claves y cantidades TEST. Las etiquetas AQ TEST no entran en el modelo.

La comparación existente `scripts/evaluation/kernel_comparison.py` acepta una
cuarta variante opcional mediante `--wind-run` y `--wind-test`, además de sus
tres parejas obligatorias sin viento. Verifica las mismas claves/targets y
configuración, y presenta beta junto con MAE, RMSE y WAPE. No selecciona
hiperparámetros con TEST. Una mejora del piloto requerirá después validación
en las particiones del protocolo completo.

## Comprobación local

```powershell
uv run pytest tests/test_convex_kernel.py tests/test_convex_kernel_optimized.py tests/test_convex_kernel_wind.py tests/test_train_convex_kernel_cli.py tests/test_convex_kernel_evaluation.py tests/test_kernel_comparison.py tests/test_gnn_cache.py tests/test_gnn_traffic_cache.py -q
```

Las pruebas incluyen dirección/signo, lotes CPU/CUDA, calma/ausencia, compatibilidad
de checkpoints, pares u/v completos, procedencia, cobertura, identidad de caché
y un entrenamiento sintético con evaluación común y etiquetas TEST alteradas.
Estos comandos no publican resultados ni lanzan tareas en la DGX.


## Comparación controlada con TRAIN 2019–2022

La opción `--train-start 2019-01-01` restringe tanto las ventanas de entrenamiento
como la media y desviación calculadas con sus celdas TRAIN. El resumen GNN
anterior se utiliza para recuperar la partición; su normalización 2015–2022
no se reutiliza al cambiar el periodo. Las fechas y este hecho quedan en config.json.

Ejecutar dos pilotos `--mode power`, uno con `--wind-mode none` y otro con
`--wind-mode directional --wind-cache cache/metraq/wind-kriging-direct-2015-2024`.
Ambos utilizan el mismo resumen de partición, semilla 312537310, batch 64,
hasta 100 épocas, paciencia 20 y validación 2023. El parámetro beta inicial sigue
en 0.01; no se cambia al mejor valor observado en el diagnóstico.

Los dos usan la misma implementación PyG. Las únicas diferencias son la entrada
de viento y su parámetro beta. No se lanzan simultáneamente en la misma GPU.

Después se comparan los checkpoints seleccionados por validación:

```powershell
uv run python scripts/evaluation/compare_convex_kernel_validation.py --without-wind-run output/convex-kernel/pilot-power-nowind-2019-20261018 --with-wind-run output/convex-kernel/pilot-power-wind-2019-20261018 --cache cache/metraq/no2-2010-2024 --wind-cache cache/metraq/wind-kriging-direct-2015-2024 --device cuda --output output/convex-kernel/comparison-power-wind-2019-20261018
```

El comparador reutiliza el recolector común de predicciones, mantiene exclusivamente
AQ TRAIN como contexto y calcula MAE, RMSE y WAPE en los mismos targets de 2023.
Valida fechas, parámetros, normalización, particiones, claves y checkpoints;
guarda comparison.csv, by-sensor.csv, predicciones de validación y summary.json.
La comparación usa validación ya empleada para seleccionar los checkpoints y
sigue siendo un solo piloto. No evalúa TEST ni publica registros W&B.
