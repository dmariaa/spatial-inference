"""Validation comparison must preserve keys and never expose TEST labels."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from click.testing import CliRunner

from metraq_gnn.data.cache import AQSensorCache
from metraq_gnn.data.graph import build_grid_graph
from metraq_gnn.data.kernel_window import CachedKernelDataset
from metraq_gnn.data.wind_cache import WindSourceCache, aq_grid_identity, file_sha256
from metraq_gnn.model import ConvexGeometricKernel

spec=importlib.util.spec_from_file_location("validation_comparison",
    Path(__file__).resolve().parents[1]/"scripts/evaluation/compare_convex_kernel_validation.py")
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_frozen_validation_comparison_is_paired_and_test_independent(tmp_path,monkeypatch):
    times=pd.date_range("2023-01-01",periods=25,freq="h").append(
        pd.date_range("2024-01-01",periods=25,freq="h")).as_unit("ns")
    values=np.broadcast_to(np.array([10.,18.,30.,np.nan],np.float32)[None,:,None],(50,4,1)).copy()
    aq=AQSensorCache(path=tmp_path,metadata=dict(grid_shape=[1,4],cell_size_m=1000),
        timestamps_ns=times.asi8,sensor_ids=np.arange(4),sensor_node_indices=np.arange(4),
        magnitude_ids=np.array([8]),values=values,availability=np.isfinite(values),
        graph=build_grid_graph({"grid":np.zeros((1,4))}))
    wind_values=np.zeros((50,4,2),np.float32)
    wind_values[:,:,0]=2.
    wind=WindSourceCache(path=tmp_path,metadata=dict(aq_grid=aq_grid_identity(aq),fingerprint="wind-id"),
        timestamps_ns=times.asi8,node_indices=np.arange(4),values=wind_values,
        availability=np.ones((50,4),bool))
    monkeypatch.setattr(comparison,"load_aq_sensor_cache",lambda _:aq)
    monkeypatch.setattr(comparison,"load_wind_source_cache",lambda _:wind)
    train=np.array([[1,0,1,0]],bool)
    val=np.array([[0,1,0,0]],bool)
    test=np.array([[0,0,0,1]],bool)
    config=dict(mode="power",seed=7,hours=24,cell_size_m=1000,epsilon_m=1.,distance_scale_m=1000.,
        source_policy="all-current-visible",implementation="pyg-vectorized",
        train_start="2019-01-01 00:00:00",train_end="2022-12-31 23:00:00",train_period="2019-2022",
        validation_period="2023",normalization_period="2019-2022",normalization_mean=[10.],
        normalization_std=[2.],train_nodes=[0,2],validation_nodes=[1],test_sensor_ids=[3],
        batch_size=2,epochs_requested=100,learning_rate=.001,early_stopping_patience=20,
        optimization_loss="mae",train_windows=35041,validation_windows=2)
    folders=[]
    for mode in ("none","directional"):
        folder=tmp_path/mode
        folder.mkdir()
        current=dict(config,wind_mode=mode,wind_cache_fingerprint="wind-id",
                     wind_train_available_fraction=1. if mode=="directional" else None)
        (folder/"config.json").write_text(json.dumps(current))
        np.savez(folder/"partition.npz",train_mask=train,validation_mask=val,test_mask=test,
                 sensor_ids=aq.sensor_ids,sensor_node_indices=aq.sensor_node_indices)
        ds=CachedKernelDataset(cache=aq,wind_cache=wind if mode=="directional" else None,
            hours=24,seed=7,context_sensor_mask=train,forbidden_context_mask=val|test,target_sensor_mask=val,
            split_start="2023-01-01",split_end="2023-12-31 23:00",
            normalization_mean=[10.],normalization_std=[2.])
        model=ConvexGeometricKernel(mode="power",wind_mode=mode,initial_wind_strength=.2).prepare_geometry(
            aq.graph.pos,[0,2])
        sample=ds[0]
        loss=float((model(sample)-sample.y)[sample.target_mask].abs().mean().detach())
        torch.save(dict(epoch=0,validation_loss=loss,model_state_dict=model.state_dict()),folder/"best-model.pt")
        (folder/"summary.json").write_text(json.dumps(dict(best_epoch=0,best_validation_mae=loss,
            history=[dict(epoch=0,validation_mae=loss)],epochs_completed=1,training_seconds=1.)))
        folders.append(folder)
    before=[file_sha256(folder/"best-model.pt") for folder in folders]
    args=["--without-wind-run",str(folders[0]),"--with-wind-run",str(folders[1]),
          "--cache",str(tmp_path),"--wind-cache",str(tmp_path),"--batch-size","2"]
    predictions=[]
    for change in (0,10000):
        aq.values[:,3]=change  # TEST station never enters validation predictions.
        aq.values[25:]=change  # All 2024 observations are excluded.
        output=tmp_path/f"comparison-{change}"
        result=CliRunner().invoke(comparison.main,args+["--output",str(output)])
        assert result.exit_code==0,result.output+repr(result.exception)
        summary=json.loads((output/"summary.json").read_text())
        assert summary["keys_and_targets_match"] and not summary["test_evaluated"]
        assert summary["validation_target_count"]==2
        assert summary["results"][1]["beta"]>0
        frame=pd.read_csv(output/"with-wind-validation-predictions.csv")
        assert set(frame.sensor_id)=={1} and pd.to_datetime(frame.window_end).dt.year.eq(2023).all()
        predictions.append(frame.prediction.to_numpy())
    np.testing.assert_array_equal(*predictions)
    assert before==[file_sha256(folder/"best-model.pt") for folder in folders]
    bad=json.loads((folders[1]/"config.json").read_text())
    bad["seed"]=999
    (folders[1]/"config.json").write_text(json.dumps(bad))
    result=CliRunner().invoke(comparison.main,args+["--output",str(tmp_path/"rejected")])
    assert result.exit_code!=0 and "protocols differ: seed" in result.output
