"""Joint meteorology/AQ PCA with fixed blocks and matched meteorology controls."""
import hashlib
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from analyze_meteorology_pca import VARIABLES,VARIANTS,fit_pca,collect_fit,overview


@click.command()
@click.option('--source',type=click.Path(exists=True,path_type=Path),required=True)
@click.option('--output',type=click.Path(path_type=Path),required=True)
@click.option('--min-rows',type=click.IntRange(22),default=168)
def main(source,output,min_rows):
    if output.exists():
        raise click.ClickException('Choose a new output directory.')
    files=sorted(source.glob('residuals_*.parquet'))
    if not files:
        raise click.ClickException('No residual datasets.')
    available=set()
    for path in files:
        available.update(pd.read_parquet(path).columns)
    all_aq=sorted(available-set(VARIABLES)-{'variant'})
    blocks={'meteo_nox':['NOX'],'meteo_nox_o3':['NOX','O3'],
            'meteo_nitrogen_o3':['NO','NO2','NOX','O3'],'meteo_all_aq':all_aq}
    output.mkdir(parents=True)
    variance,loadings,scales,coverage,stability=[],[],[],[],[]
    for path in files:
        mode,station=path.stem.split('_')[1:]
        station=int(station)
        data=pd.read_parquet(path)
        for block,aq in blocks.items():
            variables=VARIABLES+aq
            frames={v:data[data.variant.eq(v)].reindex(columns=variables).sort_index() for v in VARIANTS}
            index=frames[VARIANTS[0]].index
            assert index.is_unique and all(f.index.equals(index) for f in frames.values())
            mask=np.ones(len(index),dtype=bool)
            for frame in frames.values():
                mask&=np.isfinite(frame.to_numpy()).all(axis=1)
            selected=index[mask]
            status='eligible' if len(selected)>=min_rows else 'insufficient_complete_rows'
            models={}
            if status=='eligible':
                for variant,frame in frames.items():
                    models[(variant,'joint')]=fit_pca(frame.loc[selected],variables)
                    models[(variant,'meteorology_control')]=fit_pca(frame.loc[selected],VARIABLES)
                if any(m is None for m in models.values()):
                    status='constant_variable'
            coverage.append(dict(block=block,mode=mode,sensor_id=station,n_variables=len(variables),complete_rows=len(selected),status=status))
            if status!='eligible':
                continue
            click.echo(f'{block} {mode}: station {station}, {len(selected):,} rows, {len(variables)} variables',err=True)
            folder=output/'matched_hours'/block
            folder.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({'entry_date':selected}).to_parquet(folder/f'{mode}_{station}.parquet',index=False)
            for (variant,kind),model in models.items():
                meta=dict(block=block,kind=kind,mode=mode,sensor_id=station,variant=variant,scope='station',year=None)
                collect_fit(model,meta,variance,loadings,scales)
            for year in sorted(set(selected.year)):
                year_index=selected[selected.year==year]
                yearly={}
                if len(year_index)>=min_rows:
                    for variant,frame in frames.items():
                        yearly[(variant,'joint')]=fit_pca(frame.loc[year_index],variables)
                        yearly[(variant,'meteorology_control')]=fit_pca(frame.loc[year_index],VARIABLES)
                if not yearly or any(m is None for m in yearly.values()):
                    stability.append(dict(block=block,mode=mode,sensor_id=station,year=int(year),n_rows=len(year_index),status='insufficient_or_constant'))
                    continue
                for (variant,kind),model in yearly.items():
                    meta=dict(block=block,kind=kind,mode=mode,sensor_id=station,variant=variant,scope='station_year',year=int(year))
                    collect_fit(model,meta,variance,loadings,scales)
                    reference=models[(variant,kind)]
                    k=int(np.searchsorted(np.cumsum(reference['ratio']),0.95)+1)
                    full=k==len(model['variables'])
                    cosines=np.linalg.svd(reference['components'][:k]@model['components'][:k].T,compute_uv=False)
                    angle=float(np.rad2deg(np.arccos(np.clip(cosines,0,1))).max())
                    stability.append(dict(**meta,n_rows=len(year_index),status='eligible',reference_components=k,
                                          subspace_is_full=full,max_subspace_angle_degrees=np.nan if full else angle))
    cov=pd.DataFrame(coverage)
    cov.to_csv(output/'coverage.csv',index=False)
    if not variance:
        raise click.ClickException('No coherent blocks; coverage saved.')
    var=pd.DataFrame(variance)
    var.to_csv(output/'explained_variance.csv',index=False)
    pd.DataFrame(loadings).to_csv(output/'loadings.csv',index=False)
    pd.DataFrame(scales).to_csv(output/'standardization.csv',index=False)
    pd.DataFrame(stability).to_csv(output/'year_stability.csv',index=False)
    summary=var[var.component.eq(1)].groupby(['scope','block','kind','mode','variant']).agg(
        station_groups=('sensor_id','size'),k95_min=('components_95','min'),
        k95_median=('components_95','median'),k95_max=('components_95','max'),
        pc1_median=('explained_ratio','median'),sample_rows=('n_rows','sum')).reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    for block in blocks:
        folder=output/block
        folder.mkdir()
        for kind in ['joint','meteorology_control']:
            part=var[var.block.eq(block)&var.kind.eq(kind)]
            if len(part):
                sub=folder/kind
                sub.mkdir()
                overview(part,sub,title=f'PCA {block} — {kind}: mismas muestras por estación')
    scope=dict(source=str(source),source_scope_sha256=hashlib.sha256((source/'scope.json').read_bytes()).hexdigest(),
               meteorology=VARIABLES,aq_blocks=blocks,variants=VARIANTS,minimum_complete_rows=min_rows,
               samples='Complete cases fixed across all variants and matched meteorology control within each block/station',
               standardization='Each variable equally weighted after mean/population-std scaling per fit',
               selection='Dataset, no splits; no imputation or spatial pooling',
               caveats=['Adding AQ changes dimension and weights, so variance ratios are not directly comparable between blocks.',
                        'Nitrogen and hydrocarbon families may include related or derived quantities; joint compression is not proof of meteorology utility.',
                        'Components use AQ, so they cannot serve as meteorology-only predictors.',
                        'Interpret original/hour/month_hour separately; monthly residual also removes seasonal cycle.'])
    (output/'scope.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')
    lines=['# PCA conjunta de meteorologia y AQ\n',
           'Se usan bloques fijos para no cambiar silenciosamente contaminantes entre estaciones. Cada bloque compara originales y residuos sobre las mismas horas, y una PCA meteorologica de control sobre esas horas.\n',
           '| Bloque | Tipo | Procedencia | Version | Estaciones | k95 min / mediana / max | PC1 mediana |',
           '| --- | --- | --- | --- | ---: | --- | ---: |']
    for r in summary[summary.scope.eq('station')].itertuples():
        lines.append(f'| {r.block} | {r.kind} | {r.mode} | {r.variant} | {r.station_groups} | {r.k95_min} / {r.k95_median:g} / {r.k95_max} | {100*r.pc1_median:.1f} % |')
    lines.extend(['\nVariables por bloque:\n',json.dumps(blocks,indent=2),
                  '\nNo se rellenan ausencias. La cobertura y los IDs elegibles se guardan en coverage.csv. El bloque con todos los AQ puede tener poco o ningun soporte observado.\n',
                  'Los porcentajes corresponden a varianza estandarizada, dando el mismo peso inicial a cada variable. Anadir contaminantes cambia la dimension y el peso relativo de AQ/meteo. No basta comparar porcentajes para concluir utilidad incremental.\n',
                  'NO, NO2 y NOX, o familias de hidrocarburos, pueden compartir definicion o estar relacionados. La compresion conjunta puede proceder de AQ y no demostrar redundancia meteorologica. No se ha comprobado ninguna identidad algebraica de esas columnas.\n',
                  'Los componentes conjuntos contienen contaminacion: son exploratorios, no entradas meteorologicas independientes para un modelo predictivo.\n',
                  'Se guardan cargas, escalas, varianza anual y angulos de subespacios; no p-valores ni inferencia causal. Angulos del espacio completo quedan sin informar.\n',
                  'Reproducir en un directorio nuevo:\n\n```powershell',
                  f'.venv\\Scripts\\python.exe -B scripts/analyze_joint_pca.py --source {source.as_posix()} --output output/joint_pca_repeat --min-rows {min_rows}',
                  '```'])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    click.echo(f'COMPLETE: {cov.status.eq("eligible").sum()} coherent station/block/mode matrices')


if __name__=='__main__':
    main()
