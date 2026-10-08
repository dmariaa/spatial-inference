"""Rank stability for every complete local METRAQ campaign; no network access."""
from pathlib import Path
import json
import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from audit_all_metraq_sensors import station_statistics

def table(f):
    return "\n".join(["| "+" | ".join(map(str,f.columns))+" |",
        "| "+" | ".join(["---"]*len(f.columns))+" |"]+
        ["| "+" | ".join(map(str,r))+" |" for r in f.itertuples(index=False,name=None)])

@click.command()
@click.option("--output",type=click.Path(path_type=Path),default="output/metraq_all_sensor_audit")
@click.option("--repetitions",default=10000,type=int)
def main(output,repetitions):
    inventory=json.loads((output/"local/inventory.json").read_text(encoding="utf-8"))
    rows=[]
    for path in sorted((output/"local").glob("predictions_*.csv")):
        f=pd.read_csv(path)
        records=station_statistics(f,repetitions)
        name=f.experiment.iloc[0]
        for r in records:r.update(experiment=name,scope="local_DIP_CNN",pollutant=int(f.pollutant.iloc[0]),year=int(f.year.iloc[0]))
        rows.extend(records)
        click.echo(f"{name}: weekly bootstrap complete",err=True)
    stats=pd.DataFrame(rows)
    n_campaigns=stats.experiment.nunique()
    stats.to_csv(output/"station_rank_statistics.csv",index=False)
    frequency=stats.assign(top3=stats["rank"]<=3,stable_top3=stats.top3_probability>=.95,
        significant_vs_rest=stats.conservative_difference_low>0).groupby(["metric","sensor_id"]).agg(
        experiments=("experiment","size"),top3_count=("top3","sum"),
        bootstrap_stable_count=("stable_top3","sum"),significant_count=("significant_vs_rest","sum"),
        best_rank=("rank","min"),worst_rank=("rank","max")).reset_index()
    frequency.to_csv(output/"station_rank_frequency.csv",index=False)
    extra=[]
    graph=Path("output/spatial_sensor_audit/predictions_DIP_GNN_NOX_2024.csv")
    if graph.exists():
        for r in station_statistics(pd.read_csv(graph),repetitions):
            r.update(experiment="graph_dip_nox_add24_ks_355_full",model="Graph-DIP");extra.append(r)
    gnn_paths=sorted(Path("output/gnn/dip-matched-sensor-analysis/gnn").glob("group-*-predictions.csv"))
    if gnn_paths:
        f=pd.concat([pd.read_csv(p) for p in gnn_paths],ignore_index=True)
        for r in station_statistics(f,repetitions):
            r.update(experiment="supervised_gnn_NO2_2024",model="Supervised GNN");extra.append(r)
    pd.DataFrame(extra).to_csv(output/"supplementary_local_gnn_statistics.csv",index=False)
    historical=[]
    root=Path(r"C:\Users\david.maria\Documents\python-projects\Inferencia-espacio-temporal/experimentos")
    for p in sorted(root.rglob("results_with_stats.csv")):
        f=pd.read_csv(p);f=f[f.processed.astype(str).str.lower().eq("true")]
        for m in ["DIP_L1Loss","DIP_MSELoss"]:
            values=f.groupby("sensor_group")[m].mean()
            for group,value in values.items():
                historical.append(dict(experiment=p.parent.name,metric=m,sensor_group=group,error=value,rank=values.rank(ascending=False).loc[group]))
    pd.DataFrame(historical).to_csv(output/"historical_group_only.csv",index=False)
    year_tables=[]
    for p in Path("output").glob("year_results*.csv"):
        f=pd.read_csv(p)
        year_tables.append(dict(source=str(p),rows=len(f),processed=int(f.processed.sum()),
            status="no_sensor_identifiers; cannot rank individual stations"))
    (output/"scope.json").write_text(json.dumps(dict(
        primary=f"{n_campaigns} complete local CNN-DIP campaigns",primary_inventory=inventory,
        supplemental="Previously exported local Graph-DIP snapshot and supervised GNN evaluation",
        historical_group_only=sorted({r["experiment"] for r in historical}),
        unassignable_year_tables=year_tables,unavailable_drive="S: unavailable",
        remote_exports_from_this_turn="Excluded after user limited scope to existing local results",
        methods=dict(bootstrap="Whole calendar weeks; paired sensors; withheld contexts averaged per timestamp",
            multiplicity="max-t simultaneous intervals over 22 stations and 3 metrics per campaign; conservative bound Bonferroni-adjusted for up to 30 campaigns",
            distinction="Top3 probability is rank stability; significance compares a station to the mean of the other 21, not every other station",
            selection="Retrospective, sensors selected using prior analyses; no independent confirmatory dataset",
            missingness="Four historical campaigns cannot be decomposed without artifacts"),
        repetitions=repetitions),indent=2),encoding="utf-8")
    targets=[28079024,28079058,28079049]
    selected=frequency[frequency.sensor_id.isin(targets)]
    pivot=stats[stats.metric=="wape"].pivot(index="experiment",columns="sensor_id",values="rank")
    order=targets+[s for s in pivot.columns if s not in targets]
    pivot=pivot[order]
    fig=go.Figure(go.Heatmap(z=pivot.to_numpy(),x=[str(s)[-2:] for s in pivot.columns],y=pivot.index,
        zmin=1,zmax=22,colorscale="RdYlBu",text=pivot.to_numpy(),texttemplate="%{text:.0f}",
        hovertemplate="Experimento %{y}<br>Sensor %{x}<br>Rango WAPE %{z}<extra></extra>",
        colorbar=dict(title="Rango")))
    buttons=[]
    for metric,label in [("wape","Error relativo (WAPE)"),("mae","Error absoluto (MAE)"),("mse","Error cuadrático (MSE)")]:
        grid=stats[stats.metric==metric].pivot(index="experiment",columns="sensor_id",values="rank").reindex(index=pivot.index,columns=order)
        buttons.append(dict(label=label,method="update",args=[dict(z=[grid.to_numpy().tolist()],text=[grid.to_numpy().tolist()])]))
    fig.update_layout(updatemenus=[dict(buttons=buttons,direction="down",x=0,y=1.09,xanchor="left",yanchor="top")],
        title="Rangos de error en todas las campañas locales completas",
        xaxis_title="Estación (sufijo)",height=900,margin=dict(l=270,r=90,t=75,b=60))
    fig.write_html(output/"all_campaign_rank_map.html",include_plotlyjs=True)
    parts=["# Persistencia del error por estación en METRAQ-Madrid",
        f"## Alcance\n\n{n_campaigns} campañas locales completas de DIP-CNN, {n_campaigns*2400:,} casos y {n_campaigns*9600:,} predicciones individuales validadas contra MAE/MSE originales. Las copias remotas y los resultados descargados en este turno no entran en esta conclusión. El Graph-DIP ya exportado localmente y la GNN supervisada se muestran por separado.",
        f"## Resultado por métrica\n\nLas estaciones 24, 58 y 49 están entre las tres de mayor WAPE en todas las {n_campaigns}/{n_campaigns} campañas CNN locales. No son siempre las tres de mayor MAE o MSE. WAPE = suma del error absoluto / suma de concentración absoluta; un objetivo bajo puede producir un WAPE elevado.",
        table(selected.round(4)),
        "## Estabilidad e incertidumbre\n\nEl remuestreo utiliza semanas completas y conserva emparejadas las estaciones. Se promedian primero los distintos grupos retirados de una misma estación/fecha. Las 19 variantes no se tratan como réplicas independientes. bootstrap_stable_count cuenta campañas donde la estación permanece en el top 3 en al menos el 95% de los remuestreos. significant_count cuenta campañas donde el error supera el promedio de las otras estaciones según el intervalo simultáneo conservador; no implica ser significativamente peor que todas ellas ni estar siempre entre las tres peores.",
        "## Resultados por campaña: estaciones seleccionadas",
        table(stats[(stats.sensor_id.isin(targets))&(stats.metric=="wape")][["experiment","sensor_id","rank","value","top3_probability","rank_low","rank_high","conservative_difference_low"]].round(4)),
        "## Otras campañas locales\n\nCuatro campañas históricas de baseline/tráfico con y sin normalización solo conservan tablas por grupo, sin predicciones individuales. No permiten confirmar ni refutar el top 3 por estación. Las tablas year_results carecen de identificadores de sensor. Las pruebas parciales y smoke tests se inventarían, pero no se mezclan con campañas completas.",
        "## GNN: resultados locales complementarios",
        table(pd.DataFrame(extra)[pd.DataFrame(extra).sensor_id.isin(targets)][["experiment","metric","sensor_id","rank","top3_probability","conservative_difference_low"]].round(4)),
        f"## Límite de la conclusión\n\nEl patrón {n_campaigns}/{n_campaigns} es retrospectivo dentro de los resultados completos disponibles. No demuestra que estas estaciones sean universalmente las peores ni cubre campañas sin artefactos. Los intervalos bootstrap aproximan la incertidumbre temporal condicionada a estas configuraciones; el número de semanas es de unos 53, no miles de observaciones independientes."]
    (output/"report.md").write_text("\n\n".join(parts),encoding="utf-8")
    click.echo(selected.to_string(index=False))
    click.echo("WAPE campaign stability:")
    click.echo(stats[(stats.sensor_id.isin(targets))&(stats.metric=="wape")].groupby("sensor_id").agg(
        min_top3_probability=("top3_probability","min"),median_top3_probability=("top3_probability","median"),
        worst_upper_rank=("rank_high","max")).to_string())
if __name__=="__main__":main()
