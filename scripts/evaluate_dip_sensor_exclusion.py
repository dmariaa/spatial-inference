"""Recalculate paired local metrics after excluding selected TEST sensors."""
from pathlib import Path
import json
import click
import numpy as np
import pandas as pd

SCENARIOS={'all_sensors':[], 'without_36_27_39':[36,27,39], 'without_58_56':[58,56], 'without_all_five':[36,27,39,58,56]}
KEYS=['sensor_group','window_end','sensor_id']

def markdown(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])

@click.command()
@click.option('--audit',type=click.Path(path_type=Path),default='output/metraq_all_sensor_audit')
@click.option('--output',type=click.Path(path_type=Path),default='output/metraq_all_sensor_audit/sensor_exclusion')
@click.option('--repetitions',type=click.IntRange(min=100),default=10000)
def main(audit,output,repetitions):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    entries=[e for e in json.loads((audit/'local/inventory.json').read_text()) if e['status']=='validated']
    for entry in entries:
        stem=f"predictions_{entry['fingerprint'][:12]}.csv"
        dip=pd.read_csv(audit/'local'/stem)
        classical=pd.read_csv(audit/'classical'/stem)
        for comparator,base in classical.groupby('method'):
            f=dip.merge(base,on=KEYS,suffixes=('_dip','_base'),validate='one_to_one')
            assert len(f)==len(dip)==9600
            assert np.allclose(f.target_dip,f.target_base,rtol=3e-5,atol=3e-5)
            for scenario,excluded in SCENARIOS.items():
                retained=f[~(f.sensor_id%100).isin(excluded)].copy()
                grouped=retained.groupby(['sensor_group','window_end'])
                counts=grouped.size()
                # Equal weight per original group/window case, even after TEST size changes.
                assert len(counts)==2400
                for metric,column in [('mae','absolute_error'),('mse','squared_error'),('wape','absolute_error')]:
                    values=grouped[[column+'_dip',column+'_base']].mean()
                    if metric=='wape':
                        denominator=grouped.target_dip.apply(lambda s:s.abs().mean())
                        assert (denominator>0).all()
                        values=100*values.div(denominator,axis=0)
                    delta=values.iloc[:,0]-values.iloc[:,1]
                    weeks=pd.to_datetime(values.index.get_level_values('window_end')).to_period('W-SUN')
                    weekly=pd.DataFrame({'delta':delta.to_numpy(),'week':weeks.astype(str)}).groupby('week').delta.agg(['sum','count'])
                    rng=np.random.default_rng(20261007)
                    weights=rng.multinomial(len(weekly),np.full(len(weekly),1/len(weekly)),size=repetitions)
                    boot=(weights@weekly['sum'].to_numpy())/(weights@weekly['count'].to_numpy())
                    if metric=='wape':
                        pooled_dip=100*retained.absolute_error_dip.sum()/retained.target_dip.abs().sum()
                        pooled_base=100*retained.absolute_error_base.sum()/retained.target_base.abs().sum()
                    else:
                        pooled_dip=retained[column+'_dip'].mean(); pooled_base=retained[column+'_base'].mean()
                    rows.append(dict(experiment=entry['name'],pollutant=entry['pollutant'],year=entry['years'][0],scenario=scenario,excluded_sensors=','.join(map(str,excluded)),comparator=comparator,metric=metric,n_cases=len(counts),n_observations=len(retained),min_test_sensors=int(counts.min()),max_test_sensors=int(counts.max()),dip_error=values.iloc[:,0].mean(),baseline_error=values.iloc[:,1].mean(),delta=delta.mean(),relative_change_percent=100*(values.iloc[:,0].mean()/values.iloc[:,1].mean()-1),delta_low=np.quantile(boot,.025),delta_high=np.quantile(boot,.975),pooled_dip_error=pooled_dip,pooled_baseline_error=pooled_base,pooled_delta=pooled_dip-pooled_base))
    results=pd.DataFrame(rows)
    assert len(results)==19*4*2*3
    old=pd.read_csv(audit/'loss_attribution/aggregate_differences.csv')
    check=results[(results.scenario=='all_sensors')&(results.metric!='wape')].merge(old,on=['experiment','comparator','metric'])
    assert np.allclose(check.delta,check.net_delta,rtol=1e-8,atol=1e-8)
    results.to_csv(output/'campaign_comparisons.csv',index=False)
    summary=results.groupby(['scenario','comparator','metric']).agg(campaigns=('experiment','size'),dip_wins=('delta',lambda s:int((s<0).sum())),dip_losses=('delta',lambda s:int((s>0).sum())),wins_ci95=('delta_high',lambda s:int((s<0).sum())),losses_ci95=('delta_low',lambda s:int((s>0).sum()))).reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    original=results[results.scenario=='all_sensors'][['experiment','comparator','metric','delta']].rename(columns={'delta':'original_delta'})
    transitions=results.merge(original,on=['experiment','comparator','metric'])
    transitions['loss_to_win']=(transitions.original_delta>0)&(transitions.delta<0)
    transitions['win_to_loss']=(transitions.original_delta<0)&(transitions.delta>0)
    transitions.to_csv(output/'transitions.csv',index=False)
    report=['# Comparativa retrospectiva excluyendo sensores',
        'Solo 19 campañas locales completas de DIP-CNN. Excluir significa retirar predicciones de TEST de la evaluación; no volver a entrenar ni eliminar observaciones de entrada. Los tres métodos se evalúan sobre exactamente los mismos sensores y casos retenidos.',
        'Escenarios: todos; sin 36/27/39; sin 58/56; sin los cinco. MAE y MSE promedian primero los sensores TEST restantes de cada caso y después los 2400 casos, conservando el peso de cada grupo/ventana aunque cambie su tamaño. Las columnas pooled ofrecen la alternativa de ponderar cada observación por igual. WAPE es el promedio del WAPE por caso, en porcentaje; pooled WAPE es el cociente de sumas globales. No equivale al promedio de WAPE por estación.',
        markdown(summary),
        '## Resultados por campaña sin los cinco sensores',
        markdown(results[results.scenario=='without_all_five'][['experiment','comparator','metric','dip_error','baseline_error','delta','relative_change_percent','delta_low','delta_high']].round(4)),
        '## Límites',
        'Selección retrospectiva usando estos mismos resultados: describe sensibilidad y no demuestra superioridad generalizable. Las variantes comparten datos y no son réplicas independientes. Intervalos del 95% mediante bootstrap de semanas completas emparejadas; no corrigen comparaciones múltiples ni selección previa de sensores. No se agregan errores brutos entre contaminantes. Se mantienen todos los casos; quedan '+str(int(results[results.scenario=='without_all_five'].min_test_sensors.min()))+' sensores TEST como mínimo por caso.',
        'Reproducir: .venv\\Scripts\\python.exe scripts\\evaluate_dip_sensor_exclusion.py']
    (output/'report.md').write_text('\n\n'.join(report),encoding='utf-8')
    click.echo(summary.to_string(index=False))
    click.echo('Validado: coincidencia de objetivos, 2400 casos por campaña y reproducción exacta de MAE/MSE originales.')

if __name__=='__main__':
    main()
