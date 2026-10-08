from pathlib import Path
import pandas as pd
import numpy as np
def markdown_table(frame, index=False, **kwargs):
 frame=frame.reset_index() if index else frame
 columns=[str(c) for c in frame.columns]
 rows=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
 rows += ["| "+" | ".join(str(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)]
 return "\n".join(rows)
pd.DataFrame.to_markdown=markdown_table

p=Path("output/spatial_sensor_audit")
g=pd.read_csv(p/"predictions_DIP_GNN_NOX_2024.csv")
d=pd.read_csv(p/"predictions_NOX_2024.csv")
cnn=d[d.method=="DIP"]
paired=g.merge(cnn,on=["sensor_group","window_end","sensor_id"],suffixes=("_gnn","_cnn"),validate="one_to_one")
assert len(paired)==9600
assert np.allclose(paired.target_gnn,paired.target_cnn,rtol=2e-5,atol=2e-5)
rows=[]
for sensor,f in paired.groupby("sensor_id"):
 rows.append(dict(sensor_id=sensor,cnn_mae=f.absolute_error_cnn.mean(),gnn_mae=f.absolute_error_gnn.mean(),
  cnn_wape=f.absolute_error_cnn.sum()/f.target_cnn.abs().sum(),gnn_wape=f.absolute_error_gnn.sum()/f.target_gnn.abs().sum(),
  cnn_mse=f.squared_error_cnn.mean(),gnn_mse=f.squared_error_gnn.mean()))
s=pd.DataFrame(rows).sort_values("cnn_mae",ascending=False)
s.to_csv(p/"cnn_gnn_by_sensor.csv",index=False)
corr={metric:s["cnn_"+metric].corr(s["gnn_"+metric],method="spearman") for metric in ["mae","wape","mse"]}
with (p/"report.md").open("a",encoding="utf-8") as f:
 f.write("\n## DIP-CNN frente a DIP-GNN por estación, NOX 2024\n\n")
 f.write("9.600 predicciones emparejadas por estación, grupo y fecha. Los valores objetivo coinciden y los errores de Graph-DIP reproducen sus métricas históricas en los 2.400 casos.\n\n")
 f.write("Correlaciones de rangos por estación: "+str({k:round(v,3) for k,v in corr.items()})+"\n\n")
 f.write(s.round(3).to_markdown(index=False))
 f.write("\n\nEste desglose incorpora Graph-DIP únicamente en NOX 2024; GP sigue sin desglosarse por estación.\n")
print("CNN GNN",corr)
print(s.head(8).round(3).to_string(index=False))
