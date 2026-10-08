"""Retrospective spatial error audit using saved experiments; never trains models."""
from pathlib import Path
import json
from functools import lru_cache
import click
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from metraq_dip.utils.render_window_visualizations import _load_config, _load_window_arrays, _experiment_file, _baseline_surface
from metraq_dip.utils.sensor_error_analysis import validate_window_aggregates
from metraq_dip.tools.interpolator import KrigingInterpolator, IdwInterpolator

ROOTS = {
    "NO_2024": "experiments_NO/experiment_add24h_newnorm",
    "NO2_2024": "basic/single_channel_supervision_NO2",
    "NOX_2024": "basic/metraq_nox_add24_ks_355",
    "NOX_2023": "surface_selection_2023/ensemble_kbest_mean",
}

def _coverage_uncached(mask, point):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.nan, np.nan, 0, np.nan
    distances = np.linalg.norm(pts - point, axis=1)
    outside = np.nan
    if len(pts) >= 3:
        try:
            outside = float(Delaunay(pts).find_simplex(point) < 0)
        except QhullError:
            pass
    return distances.min(), np.sort(distances)[:3].mean(), int((distances <= 5).sum()), outside

@lru_cache(maxsize=100000)
def _cached_coverage(mask_bytes, shape, point):
    return _coverage_uncached(np.frombuffer(mask_bytes,dtype=bool).reshape(shape),np.array(point))

def coverage(mask, point):
    return _cached_coverage(np.asarray(mask,dtype=bool).tobytes(),mask.shape,tuple(point))

@click.command()
@click.option("--output", type=click.Path(path_type=Path), default="output/spatial_sensor_audit")
def main(output):
    output.mkdir(parents=True, exist_ok=True)
    cache = Path("cache/metraq/no2-2010-2024")
    ids = np.load(cache / "sensor_ids.npy")
    nodes = np.load(cache / "sensor_node_indices.npy")
    mapping = dict(zip(ids.astype(int), nodes.astype(int)))
    rows = []
    for name, relative in ROOTS.items():
        root = Path("output/experiments") / relative
        config = _load_config(root)
        expected = pd.read_csv(root / "results.csv")
        if len(expected) != 2400 or not expected.processed.all():
            raise RuntimeError(f"Incomplete experiment: {root}")
        records = []
        for i, case in enumerate(expected.itertuples(index=False)):
            arrays = _load_window_arrays(_experiment_file(root, case.sensor_group, case.time_window), config)
            if arrays["test_mask"].sum() != 4:
                raise RuntimeError("Unexpected TEST count")
            surfaces = {"DIP": arrays["DIP"]}
            with np.load(_experiment_file(root, case.sensor_group, case.time_window), allow_pickle=True) as saved:
                ensemble_train_masks = np.asarray(saved["train_mask"][:,0,-1], dtype=bool)
            pts = np.argwhere(arrays["test_mask"])
            observed_mask = arrays["train_mask"] | arrays["val_mask"]
            observed_data = np.where(arrays["train_mask"], arrays["train_data"], arrays["val_data"])
            for method, cls in (("KRG", KrigingInterpolator), ("IDW", IdwInterpolator)):
                interpolator = cls(observed_data[None], observed_mask[None])
                values = interpolator(pts[:,1].astype(float), pts[:,0].astype(float), mode="points")
                surface = np.full(arrays["test_mask"].shape, np.nan)
                surface[pts[:,0],pts[:,1]] = np.asarray(values).reshape(-1)
                surfaces[method] = surface
            for sensor in map(int, case.sensor_group.split("-")):
                r, c = divmod(mapping[sensor], arrays["test_mask"].shape[1])
                if not arrays["test_mask"][r,c]:
                    raise RuntimeError(f"Invalid sensor mapping: {sensor}")
                target = float(arrays["test_data"][r,c])
                tr = np.nanmean([coverage(mask, np.array([r,c])) for mask in ensemble_train_masks], axis=0)
                obs = coverage(arrays["train_mask"] | arrays["val_mask"], np.array([r,c]))
                for method, surface in surfaces.items():
                    pred = float(surface[r,c])
                    records.append(dict(experiment=name, pollutant=int(config["pollutants"][0]),
                        sensor_group=case.sensor_group, window_end=case.time_window, sensor_id=sensor,
                        grid_row=r, grid_column=c, method=method, target=target, prediction=pred,
                        error=pred-target, absolute_error=abs(pred-target), squared_error=(pred-target)**2,
                        train_nearest_km=tr[0], train_nearest3_km=tr[1], train_within5km=tr[2], train_outside_hull=tr[3],
                        observed_nearest_km=obs[0], observed_nearest3_km=obs[1], observed_within5km=obs[2], observed_outside_hull=obs[3]))
            if (i+1) % 600 == 0:
                click.echo(f"{name}: {i+1}/2400", err=True)
        frame = pd.DataFrame(records)
        validate_window_aggregates(frame, expected)
        rows.append(frame)
        frame.to_csv(output / f"predictions_{name}.csv", index=False)
        click.echo(f"{name}: aggregates validated", err=True)
    data = pd.concat(rows, ignore_index=True)
    summaries = []
    for keys, f in data.groupby(["experiment","sensor_id","method"]):
        summaries.append(dict(experiment=keys[0],sensor_id=keys[1],method=keys[2],
            count=len(f),groups=f.sensor_group.nunique(),grid_row=f.grid_row.iloc[0],grid_column=f.grid_column.iloc[0],
            target_mean=f.target.mean(),target_p95=f.target.quantile(.95),target_std=f.target.std(),
            mae=f.absolute_error.mean(),rmse=np.sqrt(f.squared_error.mean()),
            wape=f.absolute_error.sum()/f.target.abs().sum(),bias=f.error.mean(),
            nearest_km=f.train_nearest_km.mean(),nearest3_km=f.train_nearest3_km.mean(),
            outside_hull_fraction=f.train_outside_hull.mean()))
    sensors = pd.DataFrame(summaries)
    sensors.to_csv(output / "metrics_by_sensor.csv",index=False)
    group = data.groupby(["experiment","sensor_group","method"]).agg(mae=("absolute_error","mean"),
            mse=("squared_error","mean"),target_mean=("target","mean"),
            nearest_km=("train_nearest_km","mean")).reset_index()
    group.to_csv(output / "metrics_by_group.csv",index=False)
    comparisons=[]
    correlations=[]
    for exp, f in sensors.groupby("experiment"):
        pivot=f.pivot(index="sensor_id",columns="method",values=["mae","wape"])
        dip=f[f.method=="DIP"].set_index("sensor_id")
        for sensor in pivot.index:
            comparisons.append(dict(experiment=exp,sensor_id=sensor,
                dip_mae=pivot.loc[sensor,("mae","DIP")],
                krg_mae=pivot.loc[sensor,("mae","KRG")],idw_mae=pivot.loc[sensor,("mae","IDW")],
                dip_wape=pivot.loc[sensor,("wape","DIP")],
                dip_over_best_classical=pivot.loc[sensor,("mae","DIP")]/min(pivot.loc[sensor,("mae","KRG")],pivot.loc[sensor,("mae","IDW")])))
        for metric in ("mae","wape"):
            for feature in ("target_mean","target_p95","nearest_km","nearest3_km","outside_hull_fraction"):
                correlations.append(dict(experiment=exp,error_metric=metric,feature=feature,
                    spearman=dip[metric].corr(dip[feature],method="spearman"),n_sensors=len(dip)))
            for baseline in ("KRG","IDW"):
                correlations.append(dict(experiment=exp,error_metric=metric,feature=f"{baseline}_error",
                    spearman=pivot[(metric,"DIP")].corr(pivot[(metric,baseline)],method="spearman"),n_sensors=len(dip)))
    pd.DataFrame(comparisons).to_csv(output/"dip_vs_classical.csv",index=False)
    pd.DataFrame(correlations).to_csv(output/"sensor_correlations.csv",index=False)
    # Compare repeated station under different held-out groups on identical timestamps.
    context=[]
    diprows=data[data.method=="DIP"]
    for (exp,sensor),f in diprows.groupby(["experiment","sensor_id"]):
        groups=list(f.sensor_group.unique())
        if len(groups)!=2: continue
        a=f[f.sensor_group==groups[0]].set_index("window_end")
        b=f[f.sensor_group==groups[1]].set_index("window_end")
        if not a.index.equals(b.index): b=b.reindex(a.index)
        if not np.allclose(a.target,b.target,rtol=2e-5,atol=2e-5):
            raise RuntimeError("Repeated sensor targets differ")
        context.append(dict(experiment=exp,sensor_id=sensor,group_a=groups[0],group_b=groups[1],
            mae_a=a.absolute_error.mean(),mae_b=b.absolute_error.mean(),
            mae_ratio=max(a.absolute_error.mean(),b.absolute_error.mean())/min(a.absolute_error.mean(),b.absolute_error.mean()),
            nearest_a=a.train_nearest_km.mean(),nearest_b=b.train_nearest_km.mean()))
    pd.DataFrame(context).to_csv(output/"same_sensor_different_context.csv",index=False)
    # Target quantiles within each station remove between-station concentration differences.
    strata=[]
    for (exp,method,sensor),f in data.groupby(["experiment","method","sensor_id"]):
        f=f.copy()
        f["stratum"]=pd.qcut(f.target.rank(method="average"),4,labels=False,duplicates="drop")
        for q,g in f.groupby("stratum"):
            strata.append(dict(experiment=exp,method=method,sensor_id=sensor,quartile=int(q)+1,
                count=len(g),target_mean=g.target.mean(),mae=g.absolute_error.mean()))
    pd.DataFrame(strata).to_csv(output/"concentration_strata.csv",index=False)
    # Audit every complete local experiment at group level.
    ranking=[]
    for p in Path("output/experiments").rglob("results.csv"):
        d=pd.read_csv(p)
        if not {"processed","sensor_group","DIP_L1Loss","DIP_MSELoss"}<=set(d): continue
        if len(d)!=2400 or not d.processed.astype(str).str.lower().eq("true").all(): continue
        for metric in ("DIP_L1Loss","DIP_MSELoss"):
            values=d.groupby("sensor_group")[metric].mean()
            for sensor_group,value in values.items():
                ranking.append(dict(experiment=str(p.parent.relative_to(Path("output/experiments"))),
                    metric=metric,sensor_group=sensor_group,error=value,rank=values.rank(ascending=False).loc[sensor_group]))
    pd.DataFrame(ranking).to_csv(output/"all_experiment_group_ranks.csv",index=False)
    (output/"methodology.json").write_text(json.dumps(dict(experiments=ROOTS,
        geometry="Grid-cell Euclidean distance, 1 km cells; coverage averaged over all ensemble members, using available TRAIN cells at target hour",
        baseline_context="Kriging/IDW use TRAIN + VALIDATION; DIP optimization uses TRAIN",
        validation="All 9600 case-level MAE/MSE checked against original CSV",
        limitations=["Retrospective descriptive associations, not causal effects",
                      "Repeated sensors/groups/windows are not independent",
                      "Graph-DIP and GP not included in per-sensor analysis",
                      "Four representative experiments; remaining variants audited at group level"]),indent=2),encoding="utf-8")
    click.echo(sensors[sensors.method=="DIP"].sort_values(["experiment","mae"],ascending=[True,False]).groupby("experiment").head(5).to_string(index=False))
if __name__=="__main__":
    main()
