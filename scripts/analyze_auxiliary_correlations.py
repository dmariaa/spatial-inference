"""Descriptive dataset correlations by station/year, with exact pair support."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
METEO = {811: "wind_u", 812: "wind_v", 83: "temperature", 86: "humidity",
         87: "pressure", 88: "radiation", 89: "precipitation"}


def correlate(frame, variables, auxiliary, minimum, scope, mode, station=None, year=None):
    frame = frame.reindex(columns=variables)
    mask = frame.notna().astype("int64")
    support = mask.T.dot(mask)
    pearson = frame.corr(method="pearson", min_periods=2)
    spearman = frame.corr(method="spearman", min_periods=2)
    rows = []
    for a, b in itertools.combinations(variables, 2):
        n = int(support.loc[a, b])
        if not n:
            continue
        kind = "meteorology_meteorology" if a in auxiliary and b in auxiliary else "aq_aq" if a not in auxiliary and b not in auxiliary else "meteorology_aq"
        rows.append(dict(scope=scope,mode=mode,sensor_id=station,year=year,variable_a=a,variable_b=b,
                         relation=kind,valid_pairs=n,pearson=pearson.loc[a,b],spearman=spearman.loc[a,b],
                         sufficient_support=n>=minimum))
    return rows


def common_support(frame, auxiliary, targets, minimum, mode, station):
    rows = []
    for target in targets:
        columns = auxiliary + [target]
        complete = frame.reindex(columns=columns).dropna()
        n = len(complete)
        if n < 2:
            for variable in auxiliary:
                rows.append(dict(mode=mode,sensor_id=station,target=target,variable=variable,
                                 valid_pairs=n,pearson=np.nan,spearman=np.nan,sufficient_support=False))
            continue
        p = complete.corr(method="pearson")
        s = complete.corr(method="spearman")
        for variable in auxiliary:
            rows.append(dict(mode=mode,sensor_id=station,target=target,variable=variable,
                             valid_pairs=n,pearson=p.loc[variable,target],spearman=s.loc[variable,target],
                             sufficient_support=n>=minimum))
    return rows


def findings(summary, results, common):
    cases=[("temperature","humidity"),("temperature","O3"),("humidity","O3"),
           ("pressure","NOX"),("temperature","NOX")]
    lines=["\n## Resultados observados\n", "Coeficientes: mediana entre estaciones, no correlacion de filas agrupadas.\n",
           "| Variables | Estaciones | Pares validos brutos | Pearson | Spearman |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for a,b in cases:
        row=summary[(summary['mode']=='observed')&(summary.variable_a==a)&(summary.variable_b==b)]
        if row.empty:
            continue
        row=row.iloc[0]
        lines.append(f"| {a} / {b} | {int(row.stations)} | {int(row.raw_pairs):,} | {row.pearson_median:.3f} | {row.spearman_median:.3f} |")
    lines.extend(["\n## Estabilidad anual\n", "| Variables | Grupos estacion/ano | Spearman minimo | Spearman maximo | Positivos | Negativos |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for a,b in cases:
        rows=results[(results.scope=='station_year')&(results['mode']=='observed')&
                     (results.variable_a==a)&(results.variable_b==b)&results.sufficient_support&results.spearman.notna()]
        if len(rows):
            lines.append(f"| {a} / {b} | {len(rows)} | {rows.spearman.min():.3f} | {rows.spearman.max():.3f} | {int(rows.spearman.gt(0).sum())} | {int(rows.spearman.lt(0).sum())} |")
    nox=common[(common['mode']=='observed')&(common.target=='NOX')&common.sufficient_support]
    lines.append(f"\nLa comparacion observada sobre las siete variables y NOX simultaneamente queda restringida a "
                 f"{nox.sensor_id.nunique()} estaciones. Los coeficientes por parejas no tienen todos el mismo soporte. "
                 "Los contrastes all/observed mezclan diferencias de cobertura y procedencia; no aislan el efecto de interpolar. "
                 "La estabilidad del signo no elimina dependencia temporal, ciclos compartidos ni confusores. "
                 "No basta para concluir causalidad, redundancia total ni mejora predictiva.\n")
    return "\n".join(lines)


def overview(summary, output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    variables=list(METEO.values())+sorted((set(summary.variable_a)|set(summary.variable_b))-set(METEO.values()))
    variables=list(dict.fromkeys(variables))
    figure=make_subplots(rows=1,cols=2,subplot_titles=["Observados + interpolados","Solo observados"])
    for metric in ["spearman","pearson"]:
        for column,mode in enumerate(["all","observed"],1):
            z=np.full((len(variables),len(variables)),np.nan)
            support=np.zeros((len(variables),len(variables),2))
            for row in summary[summary['mode']==mode].itertuples():
                i,j=variables.index(row.variable_a),variables.index(row.variable_b)
                z[i,j]=z[j,i]=getattr(row,f"{metric}_median")
                support[i,j]=support[j,i]=[row.stations,row.raw_pairs]
            figure.add_trace(go.Heatmap(z=z,x=variables,y=variables,coloraxis="coloraxis",
                                       customdata=support,visible=metric=="spearman",
                                       hovertemplate="%{y} / %{x}<br>Mediana: %{z:.3f}<br>Estaciones: %{customdata[0]:.0f}<br>Pares brutos: %{customdata[1]:,.0f}<extra></extra>"),row=1,col=column)
    figure.update_layout(title="Correlaciones por estacion: mediana, 2019–2024",
                         coloraxis=dict(colorscale="RdBu_r",cmin=-1,cmax=1),height=850,width=1500,
                         updatemenus=[dict(buttons=[dict(label="Spearman",method="update",args=[{"visible":[True,True,False,False]}]),
                                                  dict(label="Pearson",method="update",args=[{"visible":[False,False,True,True]}])])])
    figure.update_xaxes(tickangle=-60,automargin=True)
    figure.update_yaxes(automargin=True)
    figure.write_html(output/"correlations_overview.html",include_plotlyjs=True)


@click.command()
@click.option("--meteorology-dir",type=click.Path(exists=True,path_type=Path),required=True)
@click.option("--output",type=click.Path(path_type=Path),required=True)
@click.option("--min-pairs",type=click.IntRange(2),default=168,
              help="Descriptive support threshold, not an independence assumption.")
def main(meteorology_dir,output,min_pairs):
    """All available AQ variables; no splits, model fitting, imputation or p-values."""
    source_scope=json.loads((meteorology_dir/"scope.json").read_text(encoding="utf-8"))
    paths=sorted(meteorology_dir.glob("meteo_*.parquet"))
    if not paths:
        raise click.ClickException("No admitted meteorology found.")
    output.mkdir(parents=True,exist_ok=True)
    cache=output/"aq_cache"
    cache.mkdir(exist_ok=True)
    from metraq_dip.data.metraq_db_legacy import metraq_db
    with metraq_db.engine.connect() as connection:
        connection.execute(text("SET SESSION max_statement_time=20"))
        connection.execute(text("START TRANSACTION READ ONLY"))
        catalog=[dict(r) for r in connection.execute(text("SELECT id,name,units,min_value,max_value FROM aq_magnitudes WHERE type='aq' ORDER BY id")).mappings()]
        connection.rollback()
        scope=dict(meteorology_scope=source_scope,aq_catalog=catalog,minimum_pairs=min_pairs,
                   sources="Admitted meteorology from DB export; AQ from MAD_merged_aq_data, is_valid=1 and catalog bounds",
                   scope="Dataset, all stations represented in meteorology; no splits",
                   modes=["all","observed"],target_provenance="Observed mode requires observed AQ as well as observed meteorology",
                   wind="u/v only; no linear correlation of circular direction and no repeated speed representation",
                   inference="Descriptive only; raw pair counts are not independent effective sample sizes; no p-values",
                   common_support="All seven meteorology variables and one AQ target, separately per station",
                   provenance="Derived wind observed only if both inputs observed")
        scope_path=output/"scope.json"
        if scope_path.exists() and json.loads(scope_path.read_text(encoding="utf-8"))!=scope:
            raise click.ClickException("Scope changed; choose a new directory.")
        scope_path.write_text(json.dumps(scope,indent=2,ensure_ascii=False),encoding="utf-8")
        for index,path in enumerate(paths,1):
            month=path.stem.removeprefix("meteo_")
            target_path=cache/f"aq_{month}.parquet"
            if target_path.exists():
                continue
            period=pd.Period(month,freq="M")
            params=dict(start=period.start_time.to_pydatetime(),end=(period+1).start_time.to_pydatetime())
            connection.execute(text("START TRANSACTION READ ONLY"))
            query="""SELECT d.sensor_id,d.entry_date,d.magnitude_id,d.value,d.is_interpolated
                FROM MAD_merged_aq_data d FORCE INDEX(idx_entry_date)
                JOIN aq_magnitudes m ON m.id=d.magnitude_id
                WHERE m.type='aq' AND d.entry_date>=:start AND d.entry_date<:end
                AND d.is_valid=1 AND d.value>=m.min_value AND d.value<=m.max_value"""
            aq=pd.DataFrame(connection.execute(text(query),params).fetchall(),
                            columns=["sensor_id","entry_date","magnitude_id","value","is_interpolated"])
            connection.rollback()
            aq.to_parquet(target_path,index=False)
            if index%12==0:
                click.echo(f"AQ cached {index}/{len(paths)} months through {month}",err=True)
    target_map={r['id']:r['name'] for r in catalog}
    if len(set(target_map.values()))!=len(target_map):
        target_map={r['id']:f"{r['name']} ({r['id']})" for r in catalog}
    auxiliary=list(METEO.values())
    all_results,all_common,coverage=[],[],[]
    for mode in ["all","observed"]:
        chunks=[]
        for path in paths:
            meteo=pd.read_parquet(path)
            stations=set(meteo.sensor_id)
            meteo=meteo[meteo.magnitude_id.isin(METEO)].copy()
            aq=pd.read_parquet(cache/f"aq_{path.stem.removeprefix('meteo_')}.parquet")
            aq=aq[aq.sensor_id.isin(stations)].copy()
            if mode=="observed":
                meteo=meteo[meteo.is_interpolated.eq(0)].copy()
                aq=aq[aq.is_interpolated.eq(0)].copy()
            meteo["variable"]=meteo.magnitude_id.map(METEO)
            aq["variable"]=aq.magnitude_id.map(target_map)
            data=pd.concat([meteo,aq],ignore_index=True)
            assert np.isfinite(data.value).all()
            assert not data.duplicated(["sensor_id","entry_date","variable"]).any()
            chunks.append(data.pivot(index=["sensor_id","entry_date"],columns="variable",values="value"))
        wide=pd.concat(chunks).sort_index()
        assert not wide.index.duplicated().any()
        targets=[name for name in target_map.values() if name in wide.columns]
        variables=auxiliary+targets
        wide=wide.reindex(columns=variables)
        for station,frame in wide.groupby(level="sensor_id"):
            click.echo(f"Correlations {mode}: station {station}",err=True)
            all_results.extend(correlate(frame,variables,auxiliary,min_pairs,"station",mode,station))
            all_common.extend(common_support(frame,auxiliary,targets,min_pairs,mode,station))
            for year,part in frame.groupby(frame.index.get_level_values("entry_date").year):
                all_results.extend(correlate(part,variables,auxiliary,min_pairs,"station_year",mode,station,year))
            for variable in variables:
                series=frame[variable].dropna()
                coverage.append(dict(mode=mode,sensor_id=station,variable=variable,
                                     valid_values=len(series),unique_values=series.nunique()))
        # Equal-time city mean is a sensitivity analysis, not a city-wide inference.
        city=wide.groupby(level="entry_date").mean()
        all_results.extend(correlate(city,variables,auxiliary,min_pairs,"hourly_network_mean",mode))
        pd.DataFrame(wide.groupby(level="entry_date").count()).to_parquet(output/f"network_source_counts_{mode}.parquet")
        del wide,chunks
    results=pd.DataFrame(all_results)
    common=pd.DataFrame(all_common)
    results.to_csv(output/"correlations.csv",index=False)
    common.to_csv(output/"common_support.csv",index=False)
    pd.DataFrame(coverage).to_csv(output/"coverage.csv",index=False)
    usable=results[(results.scope=="station")&results.sufficient_support&results.spearman.notna()]
    summary=usable.groupby(["mode","relation","variable_a","variable_b"]).agg(
        stations=("sensor_id","nunique"),raw_pairs=("valid_pairs","sum"),
        pearson_median=("pearson","median"),pearson_min=("pearson","min"),pearson_max=("pearson","max"),
        spearman_median=("spearman","median"),spearman_min=("spearman","min"),spearman_max=("spearman","max"),
        positive_stations=("spearman",lambda x:int(x.gt(0).sum())),negative_stations=("spearman",lambda x:int(x.lt(0).sum())))
    summary.to_csv(output/"station_summary.csv")
    overview(summary.reset_index(),output)
    (output/"report.md").write_text(
        "# Correlaciones meteorologia y calidad del aire\n\n"
        f"Periodo {source_scope['start']} a {source_scope['end']}; dataset completo sin splits. "
        "Pearson y Spearman por estacion y por estacion/ano. Pares exactos en estacion y timestamp; "
        "ausencias excluidas por pareja, sin imputacion ni ceros de relleno.\n\n"
        "Meteorologia: u, v, temperatura, humedad, presion, radiacion y precipitacion admitidas. "
        "AQ: todas las magnitudes disponibles con is_valid=TRUE y dentro del catalogo actual. "
        "La contaminacion sigue siendo objetivo separado, aunque se informan tambien asociaciones entre contaminantes.\n\n"
        "all incluye observados e interpolados; observed exige ambos valores observados. "
        "common_support exige las siete variables meteorologicas y un contaminante sobre las mismas horas "
        "de cada estacion. Cero soporte o constantes dan coeficientes ausentes, no cero.\n\n"
        f"Resumen por estaciones usa al menos {min_pairs} pares; es un umbral descriptivo, no un numero "
        "de observaciones independientes. Se guardan tambien los recuentos menores. "
        "No hay p-valores ni afirmaciones causales o de utilidad predictiva.\n\n"
        "hourly_network_mean promedia las estaciones disponibles por variable/hora como sensibilidad; "
        "su composicion puede cambiar y no demuestra representatividad de toda la ciudad. "
        "Se conservan los recuentos de fuentes. La dependencia espacial de interpolaciones y los ciclos "
        "diarios/estacionales pueden generar asociaciones. Los ciclos no se han retirado en esta fase.\n\n"
        "Archivos: correlations.csv, station_summary.csv, common_support.csv, coverage.csv y scope.json.\n\n"
        "Reproducir o reanudar con el mismo alcance:\n\n```powershell\n"
        f".venv\\Scripts\\python.exe -B scripts/analyze_auxiliary_correlations.py --meteorology-dir {meteorology_dir.as_posix()} "
        f"--output {output.as_posix()} --min-pairs {min_pairs}\n```\n"
        + findings(summary.reset_index(),results,common),encoding="utf-8")
    click.echo(f"COMPLETE: {len(results):,} correlations; {len(common):,} common-support rows")


if __name__=="__main__":
    main()
