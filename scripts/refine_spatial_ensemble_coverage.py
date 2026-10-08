from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_spatial_sensor_errors import ROOTS,coverage
from metraq_dip.utils.render_window_visualizations import _experiment_file
import textwrap
import json
output=Path("output/spatial_sensor_audit")
rows=[]
for name,relative in ROOTS.items():
 f=pd.read_csv(output/f"predictions_{name}.csv")
 root=Path("output/experiments")/relative
 geo=[]
 unique=f[f.method=="DIP"]
 for (group,window),case in unique.groupby(["sensor_group","window_end"],sort=False):
  with np.load(_experiment_file(root,group,window),allow_pickle=True) as a:
   masks=np.asarray(a["train_mask"][:,0,-1],bool)
  for r in case.itertuples():
   values=np.array([coverage(m,np.array([r.grid_row,r.grid_column])) for m in masks])
   mean=np.nanmean(values,axis=0)
   geo.append(dict(sensor_group=group,window_end=window,sensor_id=r.sensor_id,
    train_nearest_km=mean[0],train_nearest3_km=mean[1],train_within5km=mean[2],train_outside_hull=mean[3]))
 cols=["train_nearest_km","train_nearest3_km","train_within5km","train_outside_hull"]
 f=f.drop(columns=cols).merge(pd.DataFrame(geo),on=["sensor_group","window_end","sensor_id"],validate="many_to_one")
 f.to_csv(output/f"predictions_{name}.csv",index=False)
 rows.append(f)
 print(name,"coverage averaged over all ensemble members",flush=True)
data=pd.concat(rows,ignore_index=True)
source=Path("scripts/audit_spatial_sensor_errors.py").read_text(encoding="utf-8-sig")
tail=source[source.index("    summaries = []"):source.index('if __name__=="__main__":')]
import click
exec(textwrap.dedent(tail))
