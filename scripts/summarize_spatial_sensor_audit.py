from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def markdown_table(frame, index=False, **kwargs):
 frame=frame.reset_index() if index else frame
 columns=[str(c) for c in frame.columns]
 rows=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
 rows += ["| "+" | ".join(str(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)]
 return "\n".join(rows)
pd.DataFrame.to_markdown=markdown_table

p=Path("output/spatial_sensor_audit")
s=pd.read_csv(p/"metrics_by_sensor.csv")
c=pd.read_csv(p/"sensor_correlations.csv")
b=pd.read_csv(p/"dip_vs_classical.csv")
ctx=pd.read_csv(p/"same_sensor_different_context.csv")
q=pd.read_csv(p/"concentration_strata.csv")
parts=["# Diagnóstico espacial retrospectivo de DIP\n",'## Resultado principal\n\nLas estaciones 28079024, 28079058 y 28079049 son las tres con mayor WAPE de DIP en las cuatro campañas representativas. No siempre ocupan el mismo orden. Este patrón relativo persiste al cambiar contaminante y año, aunque WAPE no sustituye un ajuste completo por concentración y puede aumentar cuando la concentración media es baja.\n\nLa dificultad es compartida: las correlaciones de rangos de MAE por estación entre DIP y Kriging son 0,77–0,89, y entre DIP e IDW 0,74–0,86. En NOX 2024, la correlación de MAE entre DIP-CNN y DIP-GNN es 0,94.\n\nLas estaciones 24 y 58 quedan fuera de la envolvente TRAIN en todas las ventanas. Su distancia media al vecino TRAIN más próximo es aproximadamente 3,2 y 6,6 km. La estación 49 está dentro de la envolvente y tiene un vecino a 1,1–1,2 km: aislamiento y extrapolación geométrica no bastan como explicación.\n\nEl MAE de una misma estación cambia al retirar grupos diferentes: la razón mediana entre el contexto más difícil y el más fácil es 1,04–1,05; los máximos están entre 1,16 y 1,21. Son comparaciones emparejadas en fecha y objetivo, pero no aíslan causalmente qué vecino provoca el cambio.\n\nHay sobreestimación persistente en 24, 58 y 49. Los picos de concentración aumentan el error de todos los métodos. La conclusión defendible es una dificultad espacial recurrente del problema de reconstrucción con diferencias por método, no un fallo universal ni exclusivo de DIP.\n\n',
"Cuatro experimentos completos: NO, NO₂ y NOX de 2024, y NOX de 2023. 9.600 casos, 38.400 predicciones por método. DIP, Kriging e IDW. Las métricas de los tres métodos se verificaron contra los CSV originales en todos los casos.\n",
"Las distancias son aproximaciones entre centros de celdas de 1 km. La envolvente y las distancias de DIP utilizan TRAIN disponible en la hora objetivo, promediando los cinco miembros del ensemble; los baselines utilizan TRAIN + VALIDATION. Las correlaciones son descriptivas, calculadas entre 22 estaciones; no son pruebas causales ni observaciones independientes. WAPE = suma del error absoluto / suma de concentración absoluta.\n"]
for exp,f in s[s.method=="DIP"].groupby("experiment"):
 parts.append(f"\n## {exp}\n\n### Estaciones con mayor MAE\n")
 parts.append(f.sort_values("mae",ascending=False).head(8)[["sensor_id","mae","wape","target_mean","nearest_km","outside_hull_fraction"]].round(3).to_markdown(index=False))
 parts.append("\n\n### Estaciones con mayor WAPE\n")
 parts.append(f.sort_values("wape",ascending=False).head(5)[["sensor_id","mae","wape","target_mean"]].round(3).to_markdown(index=False))
 parts.append("\n\n### Asociación del error con concentración y cobertura\n")
 parts.append(c[c.experiment==exp][["error_metric","feature","spearman"]].round(3).to_markdown(index=False))
 parts.append("\n\n### Comparación con interpoladores\n")
 parts.append(b[b.experiment==exp].sort_values("dip_mae",ascending=False).head(8).round(3).to_markdown(index=False))
 parts.append("\n\n### Sensibilidad al grupo retirado\n")
 parts.append(ctx[ctx.experiment==exp].sort_values("mae_ratio",ascending=False).head(5).round(3).to_markdown(index=False))
 quart=q[q.experiment==exp].groupby(["method","quartile"]).mae.mean().unstack()
 parts.append("\n\nMAE por cuartil de concentración dentro de cada estación (promedio con igual peso por estación):\n")
 parts.append(quart.round(3).to_markdown())
 parts.append("\n")
# Geographic representation shares the grid and basemap with sensor_groups.html.
from build_spatial_error_map import build_map
build_map(p)
dip=s[s.method=="DIP"]
# Regression adjustment for concentration; log transform and standardized predictors.
regs=[]
for exp,f in dip.groupby("experiment"):
 for spatial in ["nearest_km","outside_hull_fraction"]:
  X=np.column_stack([np.log1p(f.target_mean),f[spatial].to_numpy()])
  X=(X-X.mean(axis=0))/X.std(axis=0)
  y=np.log1p(f.mae.to_numpy())
  coef=np.linalg.lstsq(np.column_stack([np.ones(len(f)),X]),y,rcond=None)[0]
  regs.append(dict(experiment=exp,spatial_feature=spatial,concentration_coefficient=coef[1],spatial_coefficient=coef[2]))
pd.DataFrame(regs).to_csv(p/"concentration_adjusted_associations.csv",index=False)
parts.extend(["\n## Ajuste descriptivo por concentración\n\nRegresión entre estaciones: log(1+MAE) ~ log(1+concentración media) + variable espacial; predictores estandarizados. Sin interpretación causal ni significación confirmatoria.\n",pd.DataFrame(regs).round(3).to_markdown(index=False),
"\n\n## Límites\n\nDIP-GNN se añade en una sección específica para NOX 2024. GP no se ha desglosado por sensor. Las 19 variantes locales se auditan por grupo en all_experiment_group_ranks.csv; el desglose individual corresponde a las cuatro campañas representativas. No se asignan nombres de barrios a partir de coordenadas de rejilla. Los valores TEST se usan únicamente para evaluación retrospectiva.\n"])
(p/"report.md").write_text("\n".join(parts),encoding="utf-8")
print(c.round(3).to_string(index=False))
print("ADJUSTED",pd.DataFrame(regs).round(3).to_string(index=False))
print("TOP SENSOR",dip.sort_values(["experiment","mae"],ascending=[True,False]).groupby("experiment").head(3)[["experiment","sensor_id","mae","wape"]].round(3).to_string(index=False))
print("CONTEXT",ctx.groupby("experiment").mae_ratio.agg(["median","max"]).round(3).to_string())
print("CLASSICAL",b.groupby("experiment").dip_over_best_classical.agg(["median","min","max"]).round(3).to_string())
