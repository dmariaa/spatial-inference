"""Standardized meteorology PCA on matched samples, by station and year."""
import hashlib
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from analyze_auxiliary_correlations import METEO

VARIANTS=['original_matched','hour','month_hour']
VARIABLES=list(METEO.values())


def fit_pca(frame,variables=None):
    variables=VARIABLES if variables is None else list(variables)
    values=frame[variables].to_numpy(dtype=float)
    mean=values.mean(axis=0)
    scale=values.std(axis=0,ddof=0)
    if not np.isfinite(values).all() or np.any(scale<=1e-12):
        return None
    standardized=(values-mean)/scale
    _,singular,components=np.linalg.svd(standardized,full_matrices=False)
    for row in components:
        if row[np.argmax(np.abs(row))]<0:
            row*=-1
    ratios=singular**2/np.sum(singular**2)
    correlation_loadings=components.T*singular/np.sqrt(len(frame))
    correlation_loadings[:,singular<=singular[0]*1e-12]=np.nan
    return dict(mean=mean,scale=scale,components=components,ratio=ratios,variables=variables,
                correlation_loadings=correlation_loadings,
                eigenvalues=singular**2/(len(frame)-1),n=len(frame))


def collect_fit(model,meta,variance,loadings,scales):
    cumulative=np.cumsum(model['ratio'])
    k95=int(np.searchsorted(cumulative,0.95)+1)
    for pc,ratio in enumerate(model['ratio'],1):
        variance.append(dict(**meta,component=pc,explained_ratio=ratio,cumulative_ratio=cumulative[pc-1],
                             n_rows=model['n'],components_95=k95))
        for j,variable in enumerate(model['variables']):
            loadings.append(dict(**meta,component=pc,variable=variable,
                                 weight=model['components'][pc-1,j],
                                 variable_pc_correlation=model['correlation_loadings'][j,pc-1]))
    for j,variable in enumerate(model['variables']):
        scales.append(dict(**meta,variable=variable,mean=model['mean'][j],std=model['scale'][j],n_rows=model['n']))


def overview(variance,output,title='PCA meteorológica: varianza acumulada por estación'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figure=make_subplots(rows=1,cols=2,subplot_titles=['Observados + interpolados','Solo observados'])
    colors=['#247b9e','#dc7a23','#8d55a2']
    labels=['Original','Residual horario','Residual mes × hora']
    for column,mode in enumerate(['all','observed'],1):
        for variant,color,label in zip(VARIANTS,colors,labels):
            data=variance[(variance['mode']==mode)&(variance['variant']==variant)&variance.scope.eq('station')]
            if data.empty:
                continue
            curve=data.groupby('component').cumulative_ratio.agg(['median','min','max'])*100
            n=data.sensor_id.nunique()
            figure.add_trace(go.Scatter(x=curve.index,y=curve['max'],line=dict(width=0),showlegend=False,
                                       hoverinfo='skip',legendgroup=variant),row=1,col=column)
            figure.add_trace(go.Scatter(x=curve.index,y=curve['min'],line=dict(width=0),fill='tonexty',
                                       opacity=0.15,fillcolor=color,showlegend=False,hoverinfo='skip',legendgroup=variant),row=1,col=column)
            figure.add_trace(go.Scatter(x=curve.index,y=curve['median'],mode='lines+markers',name=label,
                                       line=dict(color=color),legendgroup=variant,showlegend=column==1,
                                       hovertemplate=f'{label}<br>Estaciones: {n}<br>Componentes: %{{x}}<br>Varianza mediana: %{{y:.2f}} %<extra></extra>'),row=1,col=column)
        figure.add_hline(y=95,line_dash='dash',line_color='#555',row=1,col=column)
    figure.update_layout(title=title,height=600,width=1200,
                         legend=dict(orientation='h',y=-0.2))
    figure.update_xaxes(title='Número de componentes',dtick=1)
    figure.update_yaxes(title='Varianza acumulada (%)',range=[0,102])
    figure.write_html(output/'pca_variance_overview.html',include_plotlyjs=True)


@click.command()
@click.option('--source',type=click.Path(exists=True,path_type=Path),required=True)
@click.option('--output',type=click.Path(path_type=Path),required=True)
@click.option('--min-rows',type=click.IntRange(8),default=168)
def main(source,output,min_rows):
    """Seven meteorology values only; no AQ, masks, imputation or database writes."""
    if output.exists():
        raise click.ClickException('Choose a new output directory.')
    files=sorted(source.glob('residuals_*.parquet'))
    if not files:
        raise click.ClickException('No residual datasets found.')
    output.mkdir(parents=True)
    variance,loadings,scales,coverage,stability=[],[],[],[],[]
    sample_dir=output/'matched_hours'
    sample_dir.mkdir()
    for path in files:
        parts=path.stem.split('_')
        mode,station=parts[1],int(parts[2])
        data=pd.read_parquet(path)
        frames={v:data[data.variant.eq(v)][VARIABLES].sort_index() for v in VARIANTS}
        index=frames[VARIANTS[0]].index
        assert index.is_unique and all(f.index.equals(index) for f in frames.values())
        matched=np.ones(len(index),dtype=bool)
        for frame in frames.values():
            matched&=np.isfinite(frame.to_numpy()).all(axis=1)
        selected=index[matched]
        status='eligible' if len(selected)>=min_rows else 'insufficient_complete_rows'
        models={}
        if status=='eligible':
            models={v:fit_pca(f.loc[selected]) for v,f in frames.items()}
            if any(model is None for model in models.values()):
                status='constant_or_nonfinite_variable'
        coverage.append(dict(mode=mode,sensor_id=station,original_hours=len(index),complete_rows=len(selected),status=status))
        pd.DataFrame({'entry_date':selected}).to_parquet(sample_dir/f'{mode}_{station}.parquet',index=False)
        if status!='eligible':
            continue
        click.echo(f'PCA {mode}: station {station}, {len(selected):,} matched hours',err=True)
        for variant,model in models.items():
            meta=dict(mode=mode,sensor_id=station,variant=variant,scope='station',year=None)
            collect_fit(model,meta,variance,loadings,scales)
        for year in sorted(set(selected.year)):
            year_index=selected[selected.year==year]
            year_models={v:fit_pca(f.loc[year_index]) for v,f in frames.items()} if len(year_index)>=min_rows else {}
            if not year_models or any(m is None for m in year_models.values()):
                stability.append(dict(mode=mode,sensor_id=station,year=year,n_rows=len(year_index),status='insufficient_or_constant'))
                continue
            for variant,model in year_models.items():
                meta=dict(mode=mode,sensor_id=station,variant=variant,scope='station_year',year=int(year))
                collect_fit(model,meta,variance,loadings,scales)
                reference=models[variant]
                k=int(np.searchsorted(np.cumsum(reference['ratio']),0.95)+1)
                cosines=np.linalg.svd(reference['components'][:k]@model['components'][:k].T,compute_uv=False)
                angles=np.rad2deg(np.arccos(np.clip(cosines,0,1)))
                stability.append(dict(mode=mode,sensor_id=station,year=int(year),variant=variant,n_rows=len(year_index),
                                      status='eligible',reference_components=k,subspace_is_full=k==len(VARIABLES),
                                      max_subspace_angle_degrees=float(angles.max()) if k<len(VARIABLES) else np.nan))
    var=pd.DataFrame(variance)
    if var.empty:
        raise click.ClickException('No coherent seven-variable matrices available; coverage retained in memory only.')
    var.to_csv(output/'explained_variance.csv',index=False)
    pd.DataFrame(loadings).to_csv(output/'loadings.csv',index=False)
    pd.DataFrame(scales).to_csv(output/'standardization.csv',index=False)
    cov=pd.DataFrame(coverage)
    cov.to_csv(output/'coverage.csv',index=False)
    pd.DataFrame(stability).to_csv(output/'year_stability.csv',index=False)
    one=var[var.component.eq(1)]
    summary=one.groupby(['scope','mode','variant']).agg(station_groups=('sensor_id','size'),
        k95_min=('components_95','min'),k95_median=('components_95','median'),k95_max=('components_95','max'),
        pc1_median=('explained_ratio','median')).reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    scope=dict(source=str(source),source_scope_sha256=hashlib.sha256((source/'scope.json').read_bytes()).hexdigest(),
               variables=VARIABLES,variants=VARIANTS,minimum_complete_rows=min_rows,
               standardization='Per station/variant/year fit, mean and population std, no AQ or masks',
               sample_selection='Exactly the same complete rows for all three variants; no missing value filling',
               population='Dataset descriptive analysis; no splits; station-specific PCA, no spatial pooling',
               stability='Annual variance and principal angles for fixed full-period k95 subspace; PC axes can swap or rotate',
               loadings='weight is unit eigenvector coefficient; variable_pc_correlation is standardized variable-score correlation',
               limitations='Linear variance compression, not causal dimension or predictive utility; interpolations and time remain dependent')
    (output/'scope.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
    overview(var,output)
    lines=['# PCA meteorologica\n',
           'Siete variables: u, v, temperatura, humedad, presion, radiacion, precipitacion. Analisis descriptivo del dataset 2019–2024, por estacion.\n',
           'Se comparan original, residual horario y residual mes/hora sobre exactamente las mismas horas completas de cada estacion. Estandarizacion por variable; no se incluyen AQ ni mascaras, no se rellenan ausencias.\n',
           '| Procedencia | Version | Estaciones | Componentes para 95 % (min / mediana / max) | PC1 mediana |',
           '| --- | --- | ---: | --- | ---: |']
    for r in summary[summary.scope.eq('station')].itertuples():
        lines.append(f'| {r.mode} | {r.variant} | {r.station_groups} | {r.k95_min} / {r.k95_median:g} / {r.k95_max} | {100*r.pc1_median:.1f} % |')
    observed=cov[cov['mode'].eq('observed')&cov.status.eq('eligible')]
    lines.append(f"\nEstaciones con matriz observada coherente: {observed.sensor_id.tolist()}. Este subconjunto no representa por si solo toda la ciudad.\n")
    lines.extend(['La varianza acumulada del grafico es mediana entre estaciones, con intervalo minimo/maximo descriptivo. El numero de componentes es un resumen de varianza lineal, no prueba de causalidad ni de utilidad predictiva.\n',
                  'Las cargas permiten ver que variables dominan cada eje; signo fijado por el peso mayor. Los componentes de distintos anos pueden intercambiarse o rotar cuando sus autovalores son cercanos. Para estabilidad se comparan varianza y subespacios, no solo etiquetas PC.\n',
                  'Cuando el subespacio contiene los siete componentes, coincide con todo el espacio de variables: su angulo seria cero por construccion, no evidencia de estabilidad. Esos angulos no se informan.\n',
                  'Los residuos mes/hora retiran tambien estacionalidad mensual. Los ciclos se estimaron excluyendo el ano analizado; esta PCA no usa un reparto predictivo.\n',
                  'Archivos: explained_variance.csv, loadings.csv, standardization.csv, coverage.csv, year_stability.csv, matched_hours y pca_variance_overview.html.\n',
                  'Reproducir en un directorio nuevo:\n\n```powershell',
                  f'.venv\\Scripts\\python.exe -B scripts/analyze_meteorology_pca.py --source {source.as_posix()} --output output/meteorology_pca_repeat --min-rows {min_rows}',
                  '```'])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    click.echo(f'COMPLETE: {cov.status.eq("eligible").sum()} station/mode matrices, {len(var)} variance rows')


if __name__=='__main__':
    main()
