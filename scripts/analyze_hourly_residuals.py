"""Leave-one-year-out hourly cycles; raw/residual correlations on identical pairs."""
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from analyze_auxiliary_correlations import METEO, correlate


def remove_cycle(frame, variant, minimum_reference):
    """Each year's reference is estimated exclusively from the other years."""
    keys=[frame.index.hour] if variant=='hour' else [frame.index.month,frame.index.hour]
    total_sum=frame.groupby(keys).sum()
    total_count=frame.groupby(keys).count()
    residual=frame.copy()*np.nan
    refs=frame.copy()*np.nan
    patterns=[]
    for year,part in frame.groupby(frame.index.year):
        part_keys=[part.index.hour] if variant=='hour' else [part.index.month,part.index.hour]
        own_sum=part.groupby(part_keys).sum().reindex(total_sum.index,fill_value=0)
        own_count=part.groupby(part_keys).count().reindex(total_count.index,fill_value=0)
        count=total_count-own_count
        mean=(total_sum-own_sum)/count.where(count>=minimum_reference)
        lookup=pd.Index(part.index.hour) if variant=='hour' else pd.MultiIndex.from_arrays([part.index.month,part.index.hour])
        values=mean.reindex(lookup).set_axis(part.index)
        support=count.reindex(lookup).set_axis(part.index)
        residual.loc[part.index]=part-values
        refs.loc[part.index]=support
        for bin_key,row in mean.iterrows():
            bin_month,bin_hour=(None,int(bin_key)) if variant=='hour' else (int(bin_key[0]),int(bin_key[1]))
            for variable in frame.columns:
                patterns.append(dict(held_out_year=int(year),variant=variant,month=bin_month,hour=bin_hour,
                                     variable=variable,reference_count=int(count.loc[bin_key,variable]),
                                     reference_mean=row[variable]))
    return residual,refs,pd.DataFrame(patterns)


def overview(summary,output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    variables=list(METEO.values())+sorted((set(summary.variable_a)|set(summary.variable_b))-set(METEO.values()))
    variants=['original_matched','hour','month_hour']
    figure=make_subplots(rows=1,cols=3,subplot_titles=['Original: mismos pares','Residual: hora','Residual: mes × hora'])
    choices=[('spearman','observed'),('spearman','all'),('pearson','observed'),('pearson','all')]
    for choice,(metric,mode) in enumerate(choices):
        for column,variant in enumerate(variants,1):
            z=np.full((len(variables),len(variables)),np.nan)
            support=np.zeros((len(variables),len(variables),2))
            for row in summary[summary['mode']==mode].itertuples():
                i,j=variables.index(row.variable_a),variables.index(row.variable_b)
                z[i,j]=z[j,i]=getattr(row,f'{metric}_{variant}')
                support[i,j]=support[j,i]=[row.stations,row.valid_pairs]
            figure.add_trace(go.Heatmap(z=z,x=variables,y=variables,coloraxis='coloraxis',customdata=support,
                                       visible=choice==0,
                                       hovertemplate='%{y} / %{x}<br>Mediana: %{z:.3f}<br>Estaciones: %{customdata[0]:.0f}<br>Mediana de pares por estación: %{customdata[1]:,.0f}<extra></extra>'),row=1,col=column)
    buttons=[]
    for choice,(metric,mode) in enumerate(choices):
        buttons.append(dict(label=f"{metric.title()} — {'observados' if mode=='observed' else 'observados + interpolados'}",
                            method='update',args=[dict(visible=[i//3==choice for i in range(12)])]))
    figure.update_layout(title='Correlaciones antes y después de retirar ciclos — mismas estaciones y pares',
                         height=900,width=1950,coloraxis=dict(colorscale='RdBu_r',cmin=-1,cmax=1),
                         updatemenus=[dict(buttons=buttons)])
    figure.update_xaxes(tickangle=-60,automargin=True)
    figure.update_yaxes(automargin=True)
    figure.write_html(output/'residual_correlations_overview.html',include_plotlyjs=True)


@click.command()
@click.option('--source',type=click.Path(exists=True,path_type=Path),required=True)
@click.option('--meteorology-dir',type=click.Path(exists=True,path_type=Path),required=True)
@click.option('--output',type=click.Path(path_type=Path),required=True)
@click.option('--min-reference',type=click.IntRange(1),default=30)
@click.option('--min-pairs',type=click.IntRange(2),default=168)
def main(source,meteorology_dir,output,min_reference,min_pairs):
    """Uses saved dataset values; no DB access, splits or model training."""
    if output.exists():
        raise click.ClickException('Choose a new output directory.')
    scope=json.loads((source/'scope.json').read_text(encoding='utf-8'))
    target_map={r['id']:r['name'] for r in scope['aq_catalog']}
    if len(set(target_map.values()))!=len(target_map):
        target_map={r['id']:f"{r['name']} ({r['id']})" for r in scope['aq_catalog']}
    paths=sorted(meteorology_dir.glob('meteo_*.parquet'))
    results,patterns,support_records=[],[],[]
    output.mkdir(parents=True)
    for mode in ['all','observed']:
        chunks=[]
        for path in paths:
            meteo=pd.read_parquet(path)
            stations=set(meteo.sensor_id)
            meteo=meteo[meteo.magnitude_id.isin(METEO)].copy()
            aq=pd.read_parquet(source/'aq_cache'/f"aq_{path.stem.removeprefix('meteo_')}.parquet")
            aq=aq[aq.sensor_id.isin(stations)].copy()
            if mode=='observed':
                meteo=meteo[meteo.is_interpolated.eq(0)].copy()
                aq=aq[aq.is_interpolated.eq(0)].copy()
            meteo['variable']=meteo.magnitude_id.map(METEO)
            aq['variable']=aq.magnitude_id.map(target_map)
            data=pd.concat([meteo,aq],ignore_index=True)
            chunks.append(data.pivot(index=['sensor_id','entry_date'],columns='variable',values='value'))
        wide=pd.concat(chunks).sort_index()
        variables=list(METEO.values())+[n for n in target_map.values() if n in wide.columns]
        wide=wide.reindex(columns=variables)
        for station,part in wide.groupby(level='sensor_id'):
            click.echo(f'Residuals {mode}: station {station}',err=True)
            raw=part.droplevel('sensor_id')
            hourly,hour_refs,hour_patterns=remove_cycle(raw,'hour',min_reference)
            monthly,month_refs,month_patterns=remove_cycle(raw,'month_hour',min_reference)
            mask=raw.notna()&hourly.notna()&monthly.notna()
            frames={'original_matched':raw.where(mask),'hour':hourly.where(mask),'month_hour':monthly.where(mask)}
            for variable in variables:
                support_records.append(dict(mode=mode,sensor_id=station,variable=variable,
                                            original_values=int(raw[variable].count()),matched_values=int(mask[variable].sum()),
                                            unavailable_reference=int((raw[variable].notna()&~mask[variable]).sum())))
            for frame in [hour_patterns,month_patterns]:
                frame['mode']=mode
                frame['sensor_id']=station
                patterns.append(frame)
            for variant,frame in frames.items():
                entries=correlate(frame,variables,list(METEO.values()),min_pairs,'station',mode,station)
                for entry in entries:
                    entry['variant']=variant
                results.extend(entries)
                for year,year_frame in frame.groupby(frame.index.year):
                    entries=correlate(year_frame,variables,list(METEO.values()),min_pairs,'station_year',mode,station,int(year))
                    for entry in entries:
                        entry['variant']=variant
                    results.extend(entries)
            residual_export=pd.concat([frames[v].assign(variant=v) for v in frames])
            residual_export.to_parquet(output/f'residuals_{mode}_{station}.parquet')
        del wide,chunks
    result=pd.DataFrame(results)
    index=['scope','mode','sensor_id','year','variable_a','variable_b']
    assert result.groupby(index,dropna=False).valid_pairs.nunique().eq(1).all()
    result.to_csv(output/'correlations.csv',index=False)
    pattern=pd.concat(patterns,ignore_index=True)
    pattern.to_parquet(output/'reference_patterns.parquet',index=False)
    pd.DataFrame(support_records).to_csv(output/'support.csv',index=False)
    selected=result[(result.scope=='station')&result.sufficient_support&result.spearman.notna()]
    summary=selected.groupby(['mode','relation','variable_a','variable_b','variant']).agg(
        stations=('sensor_id','nunique'),raw_pairs=('valid_pairs','sum'),
        pearson_median=('pearson','median'),spearman_median=('spearman','median')).reset_index()
    summary.to_csv(output/'station_summary.csv',index=False)
    # Comparison medians use the exact same stations as well as the same pairs.
    compare=result[result.scope=='station'].pivot(index=['mode','sensor_id','relation','variable_a','variable_b','valid_pairs','sufficient_support'],columns='variant',values=['pearson','spearman'])
    comparable=compare.dropna().reset_index()
    comparable.columns=['_'.join(str(x) for x in c if str(x)) if isinstance(c,tuple) else c for c in comparable.columns]
    comparable=comparable[comparable.sufficient_support]
    comparable.to_csv(output/'matched_station_comparisons.csv',index=False)
    paired=comparable.groupby(['mode','relation','variable_a','variable_b']).median(numeric_only=True).reset_index()
    paired['stations']=comparable.groupby(['mode','relation','variable_a','variable_b']).size().to_numpy()
    paired.to_csv(output/'matched_summary.csv',index=False)
    overview(paired,output)
    scope.update(cycle_estimation='Leave one whole year out, separately per variable/station/provenance mode',
                 variants=['hour','month_hour'],minimum_reference=min_reference,minimum_pairs=min_pairs,
                 comparisons='Same station/time pairs and same stations in matched_summary',
                 limitations='No causal or predictive inference; temporal/spatial dependence remains; month_hour also removes monthly mean seasonality')
    (output/'scope.json').write_text(json.dumps(scope,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Correlaciones tras retirar ciclos horarios\n',
           'Dataset 2019–2024; estimacion fuera del ano analizado, por estacion y variable. Sin splits ni acceso nuevo a BBDD.\n',
           f'Referencia minima: {min_reference} medidas de otros anos por grupo. Comparacion sobre pares y estaciones identicos; se excluyen referencias insuficientes, sin rellenar.\n',
           'hour resta la media habitual de cada hora. month_hour resta la media de mes del ano y hora: retira tambien parte de la estacionalidad mensual. No son efectos horarios equivalentes.\n',
           '| Relacion observada | Estaciones | Spearman original | Residual hora | Residual mes/hora |',
           '| --- | ---: | ---: | ---: | ---: |']
    for a,b in [('temperature','humidity'),('temperature','O3'),('humidity','O3'),('pressure','NOX'),('temperature','NOX')]:
        row=paired[(paired['mode']=='observed')&(paired.variable_a==a)&(paired.variable_b==b)]
        if len(row):
            r=row.iloc[0]
            lines.append(f'| {a} / {b} | {int(r.stations)} | {r.spearman_original_matched:.3f} | {r.spearman_hour:.3f} | {r.spearman_month_hour:.3f} |')
    lines.extend(['\nSe guardan tambien Pearson, soporte exacto por pareja y estabilidad anual. No hay p-valores. Los pares brutos no son observaciones independientes.\n',
                  'Una reduccion es compatible con ciclos compartidos; persistencia no prueba causalidad, redundancia total ni utilidad predictiva. No se modelan tendencias interanuales ni efectos de dia de semana.\n',
                  'Los patrones guardados identifican explicitamente el ano excluido. Los datos originales y los residuales quedan separados por variant; las ausencias siguen siendo NaN.\n',
                  'Reproducir en un directorio nuevo:\n\n```powershell',
                  f'.venv\\Scripts\\python.exe -B scripts/analyze_hourly_residuals.py --source {source.as_posix()} --meteorology-dir {meteorology_dir.as_posix()} --output output/hourly_residuals_repeat --min-reference {min_reference} --min-pairs {min_pairs}',
                  '```'])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    click.echo(f'COMPLETE: {len(result):,} correlations; {len(comparable):,} matched station comparisons')


if __name__=='__main__':
    main()
