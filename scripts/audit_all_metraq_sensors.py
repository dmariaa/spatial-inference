"""Inventory and extract per-station DIP errors, without training or changing experiments."""
from pathlib import Path
import hashlib,json,sys
import click
import numpy as np
import pandas as pd
import yaml

def fingerprint(frame):
    columns=["time_window","sensor_group","DIP_L1Loss","DIP_MSELoss"]
    f=frame[columns].sort_values(columns[:2]).copy()
    f[columns[2:]]=f[columns[2:]].round(8)
    return hashlib.sha256(f.to_csv(index=False,float_format="%.8f").encode()).hexdigest()

def inventory(root):
    entries=[]
    for p in sorted(root.rglob("results.csv")):
        if any(part in {".venv",".git","site","node_modules"} for part in p.parts): continue
        cfgpath=p.parent/"config.yaml"
        if not cfgpath.exists(): continue
        cfg=yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
        dataset=cfg.get("aq_dataset","metraq")
        if dataset.lower()!="metraq": continue
        f=pd.read_csv(p)
        if not {"time_window","sensor_group","processed","DIP_L1Loss","DIP_MSELoss"}<=set(f):continue
        done=f[f.processed.astype(str).str.lower().eq("true")].copy()
        complete=len(done)==len(f) and len(done)>=200
        entries.append(dict(source=str(p.parent),name=p.parent.name,dataset=dataset,
            pollutant=int(cfg["pollutants"][0]),rows=len(f),processed=len(done),
            artifacts=len(list(p.parent.glob("exp_*.npz"))),complete=complete,
            model=cfg.get("surface_optimizer","dip"),fingerprint=fingerprint(done),
            years=sorted(pd.to_datetime(done.time_window).dt.year.unique().tolist()) if len(done) else [],
            config=cfg))
    return entries

def extract(entry,mapping):
    root=Path(entry["source"]);cfg=entry["config"]
    expected=pd.read_csv(root/"results.csv")
    expected=expected[expected.processed.astype(str).str.lower().eq("true")]
    records=[]
    for case in expected.itertuples(index=False):
        ts=pd.Timestamp(case.time_window)
        path=root/f"exp_{case.sensor_group}_{ts.strftime('%Y%m%dT%H%M%S')}.npz"
        with np.load(path,allow_pickle=True) as a:
            target=np.asarray(a["test_data"][0,0,-1],float)
            mask=np.asarray(a["test_mask"][0,0,-1],bool)
            surface=np.asarray(a["train_output"],float)
            if cfg.get("normalize",False):
                mean,std=a["normalization_stats"].item()[int(cfg["pollutants"][0])]
                target=target*(std+1e-6)+mean
                surface=np.asarray(a["train_output_real"],float) if "train_output_real" in a else surface*(std+1e-6)+mean
            errors=[]
            sensors=list(map(int,case.sensor_group.split("-")))
            if mask.sum()!=len(sensors):raise RuntimeError(f"TEST count mismatch: {path}")
            for sensor in sensors:
                r,c=divmod(mapping[sensor],mask.shape[1])
                if not mask[r,c]:raise RuntimeError(f"Grid mismatch: {sensor} {path}")
                truth=float(target[r,c]);pred=float(surface[r,c]);err=pred-truth
                errors.append(err)
                records.append(dict(experiment=entry["name"],fingerprint=entry["fingerprint"],
                    pollutant=entry["pollutant"],year=ts.year,model=entry["model"],
                    sensor_group=case.sensor_group,window_end=case.time_window,sensor_id=sensor,
                    target=truth,prediction=pred,error=err,absolute_error=abs(err),squared_error=err*err))
            if not np.isclose(np.mean(np.abs(errors)),case.DIP_L1Loss,rtol=3e-5,atol=3e-5):
                raise RuntimeError(f"MAE mismatch: {path}")
            if not np.isclose(np.mean(np.square(errors)),case.DIP_MSELoss,rtol=3e-5,atol=3e-5):
                raise RuntimeError(f"MSE mismatch: {path}")
    return pd.DataFrame(records)


def station_statistics(frame, repetitions=10000):
    # Average the different withheld-group contexts before weighting each timestamp.
    f=frame.copy()
    f["target_abs"]=f.target.abs()
    averaged=f.groupby(["window_end","sensor_id"])[["absolute_error","squared_error","target_abs"]].mean().reset_index()
    averaged["week"]=pd.to_datetime(averaged.window_end).dt.to_period("W-SUN").astype(str)
    sensors=sorted(averaged.sensor_id.unique())
    weeks=sorted(averaged.week.unique())
    sums=averaged.groupby(["week","sensor_id"])[["absolute_error","squared_error","target_abs"]].sum()
    count=averaged.groupby(["week","sensor_id"]).size().unstack().reindex(index=weeks,columns=sensors).to_numpy()
    if not np.all(count==count[:,[0]]):
        raise RuntimeError("Incomplete paired sensor support; cannot use shared bootstrap")
    arrays={k:sums[k].unstack().reindex(index=weeks,columns=sensors).to_numpy() for k in sums.columns}
    rng=np.random.default_rng(20261007)
    weights=rng.multinomial(len(weeks),np.full(len(weeks),1/len(weeks)),size=repetitions)
    counts=weights@count
    errors=weights@arrays["absolute_error"]
    squares=weights@arrays["squared_error"]
    targets=weights@arrays["target_abs"]
    boots={"mae":errors/counts,"mse":squares/counts,"wape":errors/targets}
    estimates={"mae":arrays["absolute_error"].sum(axis=0)/count.sum(axis=0),
        "mse":arrays["squared_error"].sum(axis=0)/count.sum(axis=0),
        "wape":arrays["absolute_error"].sum(axis=0)/arrays["target_abs"].sum(axis=0)}
    deviations={};ses={};deltas={}
    for metric,b in boots.items():
        delta=estimates[metric]-(estimates[metric].sum()-estimates[metric])/(len(sensors)-1)
        db=b-(b.sum(axis=1,keepdims=True)-b)/(len(sensors)-1)
        se=db.std(axis=0,ddof=1)
        deviations[metric]=(db-delta)/np.maximum(se,1e-12)
        ses[metric]=se;deltas[metric]=delta
    max_t=np.max(np.abs(np.concatenate(list(deviations.values()),axis=1)),axis=1)
    cutoff=float(np.quantile(max_t,.95))
    conservative_cutoff=float(np.quantile(max_t,1-.05/30))
    records=[]
    for metric,b in boots.items():
        point_ranks=np.argsort(np.argsort(-estimates[metric]))+1
        ranks=np.argsort(np.argsort(-b,axis=1),axis=1)+1
        for j,sensor in enumerate(sensors):
            low=deltas[metric][j]-cutoff*ses[metric][j]
            high=deltas[metric][j]+cutoff*ses[metric][j]
            conservative_low=deltas[metric][j]-conservative_cutoff*ses[metric][j]
            records.append(dict(sensor_id=int(sensor),metric=metric,value=float(estimates[metric][j]),
                rank=int(point_ranks[j]),top3_probability=float((ranks[:,j]<=3).mean()),
                rank_low=float(np.quantile(ranks[:,j],.025)),rank_high=float(np.quantile(ranks[:,j],.975)),
                difference_vs_other_stations=float(deltas[metric][j]),
                simultaneous_difference_low=float(low),simultaneous_difference_high=float(high),
                conservative_difference_low=float(conservative_low),
                n_stations=len(sensors),n_weeks=len(weeks),n_windows=averaged.window_end.nunique(),
                bootstrap_repetitions=repetitions))
    return records

@click.command()
@click.option("--root",multiple=True,type=click.Path(path_type=Path),required=True)
@click.option("--cache",type=click.Path(path_type=Path),required=True)
@click.option("--output",type=click.Path(path_type=Path))
@click.option("--stream",is_flag=True)
@click.option("--inventory-only",is_flag=True)
@click.option("--aggregate",is_flag=True)
@click.option("--repetitions",default=10000,type=int)
def main(root,cache,output,stream,inventory_only,aggregate,repetitions):
    entries=[]
    for r in root: entries.extend(inventory(r))
    if inventory_only:
        click.echo(json.dumps(entries));return
    ids=np.load(cache/"sensor_ids.npy").astype(int)
    nodes=np.load(cache/"sensor_node_indices.npy").astype(int)
    mapping=dict(zip(ids,nodes))
    if output:output.mkdir(parents=True,exist_ok=True)
    if stream:
        click.echo(json.dumps(dict(type="inventory",entries=entries)))
    seen=set()
    for entry in entries:
        if not entry["complete"]:
            entry["status"]="partial_or_smoke";continue
        if entry["fingerprint"] in seen:
            entry["status"]="duplicate";continue
        seen.add(entry["fingerprint"])
        try:
            frame=extract(entry,mapping)
            entry["status"]="validated"
            if stream:
                if aggregate:
                    statistics=station_statistics(frame,repetitions)
                    click.echo(json.dumps(dict(type="statistics",fingerprint=entry["fingerprint"],
                        experiment=entry["name"],pollutant=entry["pollutant"],years=entry["years"],model=entry["model"],records=statistics)))
                else:
                    click.echo(json.dumps(dict(type="predictions",fingerprint=entry["fingerprint"],records=frame.to_dict("records"))))
            if output:frame.to_csv(output/f"predictions_{entry['fingerprint'][:12]}.csv",index=False)
            click.echo(f"{entry['name']}: {len(frame)} sensor predictions validated",err=True)
        except Exception as e:
            entry["status"]="failed";entry["error"]=str(e)
            click.echo(f"{entry['name']}: {e}",err=True)
    if output:(output/"inventory.json").write_text(json.dumps(entries,indent=2),encoding="utf-8")
    if stream:click.echo(json.dumps(dict(type="final_inventory",entries=entries)))
if __name__=="__main__":main()
