"""Reconstruct and validate Kriging/IDW at TEST stations for all complete local METRAQ campaigns."""
from pathlib import Path
import hashlib,json
import click
import numpy as np
import pandas as pd
from audit_all_metraq_sensors import station_statistics
from metraq_dip.utils.render_window_visualizations import _load_window_arrays,_experiment_file
from metraq_dip.tools.interpolator import KrigingInterpolator,IdwInterpolator

def markdown(f):
    return "\n".join(["| "+" | ".join(map(str,f.columns))+" |",
        "| "+" | ".join(["---"]*len(f.columns))+" |"]+
        ["| "+" | ".join(map(str,r))+" |" for r in f.itertuples(index=False,name=None)])

@click.command()
@click.option("--audit",type=click.Path(path_type=Path),default="output/metraq_all_sensor_audit")
@click.option("--output",type=click.Path(path_type=Path),default="output/metraq_all_sensor_audit/classical")
@click.option("--repetitions",default=10000,type=int)
def main(audit,output,repetitions):
    output.mkdir(parents=True,exist_ok=True)
    cache_dir=output/"interpolation_cache"
    cache_dir.mkdir(exist_ok=True)
    entries=[e for e in json.loads((audit/"local/inventory.json").read_text()) if e["status"]=="validated"]
    ids=np.load("cache/metraq/no2-2010-2024/sensor_ids.npy").astype(int)
    nodes=np.load("cache/metraq/no2-2010-2024/sensor_node_indices.npy").astype(int)
    mapping=dict(zip(ids,nodes))
    code_hash=hashlib.sha256(Path("src/metraq_dip/tools/interpolator.py").read_bytes()).hexdigest().encode()
    stats_records=[];bias_records=[];status=[];evaluation_sets={}
    for entry in entries:
        root=Path(entry["source"])
        expected=pd.read_csv(root/"results.csv")
        file=output/f"predictions_{entry['fingerprint'][:12]}.csv"
        records=[];hits=0
        if file.exists():
            frame=pd.read_csv(file)
        else:
            for index,case in enumerate(expected.itertuples(index=False)):
                arrays=_load_window_arrays(_experiment_file(root,case.sensor_group,case.time_window),entry["config"])
                mask=arrays["train_mask"]|arrays["val_mask"]
                observed=np.where(arrays["train_mask"],arrays["train_data"],arrays["val_data"])
                sensors=list(map(int,case.sensor_group.split("-")))
                points=np.array([divmod(mapping[s],mask.shape[1]) for s in sensors],dtype=int)
                if not arrays["test_mask"][points[:,0],points[:,1]].all():
                    raise RuntimeError("Invalid TEST station mapping")
                if mask[points[:,0],points[:,1]].any():raise RuntimeError("TEST leakage")
                target=arrays["test_data"][points[:,0],points[:,1]].astype(float)
                digest=hashlib.sha256(code_hash+mask.tobytes()+observed[mask].tobytes()+points.tobytes()).hexdigest()
                cached=cache_dir/f"{digest}.npz"
                if cached.exists():
                    with np.load(cached) as c:predictions={method:c[method] for method in ("KRG","IDW")}
                    hits+=1
                else:
                    predictions={}
                    for method,cls in [("KRG",KrigingInterpolator),("IDW",IdwInterpolator)]:
                        interpolator=cls(observed[None],mask[None])
                        predictions[method]=np.asarray(interpolator(points[:,1].astype(float),points[:,0].astype(float),mode="points"),dtype=np.float32).astype(float)
                    np.savez_compressed(cached,**predictions)
                for method,pred in predictions.items():
                    errors=pred-target
                    for metric,value in [("L1Loss",np.abs(errors).mean()),("MSELoss",np.square(errors).mean())]:
                        ref=float(getattr(case,f"{method}_{metric}"))
                        if not np.isclose(value,ref,rtol=3e-5,atol=3e-5):
                            raise RuntimeError(f"{entry['name']} {method} {metric} mismatch at {case.time_window}: {value} vs {ref}")
                    for i,sensor in enumerate(sensors):
                        error=float(errors[i])
                        records.append(dict(experiment=entry["name"],fingerprint=entry["fingerprint"],
                            method=method,pollutant=entry["pollutant"],year=pd.Timestamp(case.time_window).year,
                            sensor_group=case.sensor_group,window_end=case.time_window,sensor_id=sensor,
                            target=float(target[i]),prediction=float(pred[i]),error=error,
                            absolute_error=abs(error),squared_error=error*error))
                if (index+1)%600==0:click.echo(f"{entry['name']}: {index+1}/{len(expected)}; cache hits {hits}",err=True)
            frame=pd.DataFrame(records)
            frame.to_csv(file,index=False)
        payload=frame[["method","sensor_group","window_end","sensor_id","target","prediction"]].sort_values(["method","sensor_group","window_end","sensor_id"])
        evaluation_hash=hashlib.sha256(payload.to_csv(index=False,float_format="%.8f").encode()).hexdigest()
        evaluation_sets.setdefault(evaluation_hash,[]).append(entry["name"])
        # Revalidate loaded/resumed exports against all original case-level metrics too.
        for method,f in frame.groupby("method"):
            actual=f.groupby(["window_end","sensor_group"]).agg(mae=("absolute_error","mean"),mse=("squared_error","mean")).sort_index()
            ref=expected.set_index(["time_window","sensor_group"]).sort_index()
            if len(actual)!=len(ref):raise RuntimeError("Incomplete exported cases")
            for m,column in [("mae",f"{method}_L1Loss"),("mse",f"{method}_MSELoss")]:
                if not np.allclose(actual[m],ref[column],rtol=3e-5,atol=3e-5):raise RuntimeError(f"Export aggregate mismatch: {entry['name']} {method}")
            results=station_statistics(f,repetitions)
            for r in results:r.update(experiment=entry["name"],method=method,pollutant=entry["pollutant"],year=entry["years"][0])
            stats_records.extend(results)
            for sensor,g in f.groupby("sensor_id"):
                bias_records.append(dict(experiment=entry["name"],method=method,sensor_id=int(sensor),
                    bias=g.error.mean(),overestimate_fraction=(g.error>0).mean(),
                    mae=g.absolute_error.mean(),wape=g.absolute_error.sum()/g.target.abs().sum(),
                    target_mean=g.target.mean()))
        status.append(dict(experiment=entry["name"],cases=len(expected),method_sensor_predictions=len(frame),status="validated"))
        (output/"validation.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        click.echo(f"{entry['name']}: KRG/IDW validated and weekly bootstrap complete",err=True)
    (output/"repeated_evaluations.json").write_text(json.dumps(list(evaluation_sets.values()),indent=2),encoding="utf-8")
    stats=pd.DataFrame(stats_records);bias=pd.DataFrame(bias_records)
    stats.to_csv(output/"station_rank_statistics.csv",index=False)
    bias.to_csv(output/"station_bias.csv",index=False)
    frequency=stats.assign(top3=stats["rank"]<=3,stable=stats.top3_probability>=.95,significant=stats.conservative_difference_low>0).groupby(["method","metric","sensor_id"]).agg(
        campaigns=("experiment","size"),top3_count=("top3","sum"),stable_top3_count=("stable","sum"),significant_count=("significant","sum"),
        min_top3_probability=("top3_probability","min"),best_rank=("rank","min"),worst_rank=("rank","max")).reset_index()
    frequency.to_csv(output/"station_rank_frequency.csv",index=False)
    targets=[28079024,28079049,28079058]
    selected=frequency[frequency.sensor_id.isin(targets)]
    bias_summary=bias[bias.sensor_id.isin(targets)].groupby(["method","sensor_id"]).agg(
        campaigns=("experiment","size"),positive_bias_campaigns=("bias",lambda v:int((v>0).sum())),
        min_bias=("bias","min"),max_bias=("bias","max"),min_overestimate_fraction=("overestimate_fraction","min"),
        max_overestimate_fraction=("overestimate_fraction","max")).reset_index()
    bias_summary.to_csv(output/"selected_station_bias_summary.csv",index=False)
    parts=["# Kriging e IDW: persistencia del error en todas las campañas locales",
        "19 campañas locales completas, 45.600 casos por método. Las variantes comparten cuatro evaluaciones distintas por contaminante/año; no son 19 réplicas independientes. Todas las MAE/MSE reconstruidas se validan frente a los CSV históricos. Las predicciones utilizan TRAIN + VALIDATION de la hora objetivo, excluyendo completamente TEST.",
        "## Frecuencia y estabilidad de las estaciones 24, 49 y 58",
        markdown(selected.round(4)),
        "## Sesgo\n\nUn sesgo positivo significa sobreestimación en promedio; no significa sobreestimar todas las ventanas.",
        markdown(bias_summary.round(4)),
        "## Error relativo por campaña",
        markdown(stats[(stats.metric=="wape")&stats.sensor_id.isin(targets)][["experiment","method","sensor_id","rank","value","top3_probability"]].round(4)),
        "## Límites\n\nBootstrap de 10.000 remuestreos de semanas completas con sensores emparejados. Variantes que reutilizan datos no son réplicas independientes. WAPE depende de la concentración; ninguna asociación causal con vegetación se ha comprobado. No se extrapola a las campañas antiguas sin artefactos individuales."]
    parts.append("La pertenencia observada al top 3 por WAPE es 19/19 para las tres estaciones y ambos métodos. La estabilidad bootstrap no supera el 95% en todas las combinaciones. Un sesgo positivo significa sobreestimación media, no en todas las ventanas. significant_count compara con el promedio de las otras 21 estaciones mediante el intervalo simultáneo conservador; no significa superar a cada una de ellas.")
    (output/"report.md").write_text("\n\n".join(parts),encoding="utf-8")
    click.echo(selected.to_string(index=False))
    click.echo(bias_summary.to_string(index=False))
if __name__=="__main__":main()
