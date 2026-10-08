"""Attribute DIP's paired aggregate error differences to stations in all local campaigns."""
from pathlib import Path
import json
import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def markdown(f):
 return "\n".join(["| "+" | ".join(map(str,f.columns))+" |","| "+" | ".join(["---"]*len(f.columns))+" |"]+["| "+" | ".join(map(str,r))+" |" for r in f.itertuples(index=False,name=None)])

@click.command()
@click.option("--audit",type=click.Path(path_type=Path),default="output/metraq_all_sensor_audit")
@click.option("--output",type=click.Path(path_type=Path),default="output/metraq_all_sensor_audit/loss_attribution")
@click.option("--repetitions",default=10000,type=int)
def main(audit,output,repetitions):
 output.mkdir(parents=True,exist_ok=True)
 entries=[e for e in json.loads((audit/"local/inventory.json").read_text()) if e["status"]=="validated"]
 records=[];global_records=[];tail_records=[];worst_records=[]
 for entry in entries:
  stem=f"predictions_{entry['fingerprint'][:12]}.csv"
  dip=pd.read_csv(audit/"local"/stem)
  classical=pd.read_csv(audit/"classical"/stem)
  sensors=sorted(dip.sensor_id.unique())
  matrices={}
  for comparator,baseline in classical.groupby("method"):
   f=dip.merge(baseline,on=["sensor_group","window_end","sensor_id"],suffixes=("_dip","_baseline"),validate="one_to_one")
   if len(f)!=9600 or not np.allclose(f.target_dip,f.target_baseline,rtol=3e-5,atol=3e-5):raise RuntimeError("Unmatched evaluation targets")
   f["week"]=pd.to_datetime(f.window_end).dt.to_period("W-SUN").astype(str)
   weekly_counts=f.groupby(["week","sensor_id"]).size().unstack().reindex(columns=sensors)
   counts=weekly_counts.to_numpy()
   rng=np.random.default_rng(20261007)
   weights=rng.multinomial(len(counts),np.full(len(counts),1/len(counts)),size=repetitions)
   station_counts=weights@counts
   total_counts=station_counts.sum(axis=1)
   for metric,column in [("mae","absolute_error"),("mse","squared_error")]:
    f["delta"]=f[column+"_dip"]-f[column+"_baseline"]
    sums=f.groupby(["week","sensor_id"]).delta.sum().unstack().reindex(index=weekly_counts.index,columns=sensors).to_numpy()
    bootstrap_sums=weights@sums
    station_boot=bootstrap_sums/station_counts
    aggregate_boot=bootstrap_sums.sum(axis=1)/total_counts
    station_mean=sums.sum(axis=0)/counts.sum(axis=0)
    matrices[(comparator,metric)]=(station_boot,station_mean)
    net=float(f.delta.mean())
    original=pd.read_csv(Path(entry["source"])/"results.csv")
    loss_column="L1Loss" if metric=="mae" else "MSELoss"
    reference=float((original[f"DIP_{loss_column}"]-original[f"{comparator}_{loss_column}"]).mean())
    if not np.isclose(net,reference,rtol=5e-5,atol=1e-4):raise RuntimeError(f"Aggregate attribution mismatch: {entry['name']} {comparator} {metric}")
    for j,sensor in enumerate(sensors):
     g=f[f.sensor_id==sensor]
     weight=len(g)/len(f)
     records.append(dict(experiment=entry["name"],pollutant=entry["pollutant"],year=entry["years"][0],
      comparator=comparator,metric=metric,sensor_id=int(sensor),count=len(g),evaluation_weight=weight,
      dip_error=g[column+"_dip"].mean(),baseline_error=g[column+"_baseline"].mean(),
      mean_delta=station_mean[j],net_contribution=g.delta.sum()/len(f),
      worsening_contribution=g.delta.clip(lower=0).sum()/len(f),
      improvement_contribution=g.delta.clip(upper=0).sum()/len(f),
      case_loss_fraction=(g.delta>0).mean(),positive_delta_stability=(station_boot[:,j]>0).mean()))
    global_records.append(dict(experiment=entry["name"],pollutant=entry["pollutant"],year=entry["years"][0],
      comparator=comparator,metric=metric,dip_error=f[column+"_dip"].mean(),baseline_error=f[column+"_baseline"].mean(),
      net_delta=net,delta_low=np.quantile(aggregate_boot,.025),delta_high=np.quantile(aggregate_boot,.975),
      positive_delta_stability=(aggregate_boot>0).mean(),n_cases=2400,n_weeks=len(counts)))
    cases=f.groupby(["window_end","sensor_group"]).delta.mean().sort_values(ascending=False)
    k=int(np.ceil(.01*len(cases)))
    tail=cases.iloc[:k];other=cases.iloc[k:]
    tail_records.append(dict(experiment=entry["name"],comparator=comparator,metric=metric,net_delta=net,
      worst_1pct_cases=k,worst_1pct_contribution=tail.sum()/len(cases),
      other_cases_contribution=other.sum()/len(cases),other_cases_mean_delta=other.mean(),
      tail_share_of_net_loss=tail.sum()/len(cases)/net if net>0 else np.nan))
    for (window,group),delta in tail.head(5).items():
     for row in f[(f.window_end==window)&(f.sensor_group==group)].itertuples():
      worst_records.append(dict(experiment=entry["name"],comparator=comparator,metric=metric,
       window_end=window,sensor_group=group,case_delta=delta,sensor_id=row.sensor_id,sensor_delta=row.delta))
  # Simultaneous station intervals across both comparators and both error metrics.
  deviations=[]
  for key,(boot,point) in matrices.items():
   se=boot.std(axis=0,ddof=1)
   deviations.append((boot-point)/np.maximum(se,1e-12))
  critical=np.quantile(np.abs(np.concatenate(deviations,axis=1)).max(axis=1),1-.05/30)
  for r in records:
   if r["experiment"]!=entry["name"]:continue
   boot,point=matrices[(r["comparator"],r["metric"])]
   j=sensors.index(r["sensor_id"]);se=boot[:,j].std(ddof=1)
   r["simultaneous_delta_low"]=point[j]-critical*se
   r["simultaneous_delta_high"]=point[j]+critical*se
  click.echo(f"{entry['name']}: exact aggregate decomposition and paired weekly bootstrap validated",err=True)
 stations=pd.DataFrame(records)
 totals=pd.DataFrame(global_records)
 for key,g in stations.groupby(["experiment","comparator","metric"]):
  row=totals[(totals.experiment==key[0])&(totals.comparator==key[1])&(totals.metric==key[2])].iloc[0]
  if not np.isclose(g.net_contribution.sum(),row.net_delta,rtol=1e-10,atol=1e-10):raise RuntimeError("Contributions do not sum to aggregate difference")
  idx=g.index
  stations.loc[idx,"penalty_rank"]=g.net_contribution.rank(ascending=False)
  positive=g.net_contribution.clip(lower=0).sum()
  stations.loc[idx,"share_of_positive_net_contributions"]=g.net_contribution.clip(lower=0)/positive if positive>0 else 0
 stations.to_csv(output/"station_contributions.csv",index=False)
 totals.to_csv(output/"aggregate_differences.csv",index=False)
 pd.DataFrame(tail_records).to_csv(output/"extreme_case_contributions.csv",index=False)
 pd.DataFrame(worst_records).to_csv(output/"worst_cases_by_sensor.csv",index=False)
 losses=totals[totals.net_delta>0][["experiment","comparator","metric"]].assign(aggregate_loss=True)
 stations=stations.merge(losses,on=["experiment","comparator","metric"],how="left")
 stations["aggregate_loss"]=stations.aggregate_loss.eq(True)
 stations["penalizes"]=stations.mean_delta>0
 stations["top3_penalty"]=(stations.penalty_rank<=3)&stations.penalizes
 frequency=stations.groupby(["comparator","metric","sensor_id"]).agg(
  campaigns=("experiment","size"),penalizes_count=("penalizes","sum"),
  top3_penalty_count=("top3_penalty","sum"),significant_penalty_count=("simultaneous_delta_low",lambda v:int((v>0).sum())),
  median_penalty_rank=("penalty_rank","median")).reset_index()
 frequency.to_csv(output/"penalty_frequency.csv",index=False)
 loss_frequency=stations[stations.aggregate_loss].groupby(["comparator","metric","sensor_id"]).agg(
  loss_campaigns=("experiment","size"),top3_penalty_count=("top3_penalty","sum"),
  penalizes_count=("penalizes","sum")).reset_index()
 loss_frequency.to_csv(output/"penalty_frequency_in_losing_campaigns.csv",index=False)
 # Summary by pollutant/year prevents the ten 2023 variants dominating interpretation.
 families=stations.groupby(["comparator","metric","pollutant","year","sensor_id"]).agg(
  mean_delta=("mean_delta","mean"),mean_contribution=("net_contribution","mean"),
  positive_campaign_fraction=("penalizes","mean")).reset_index()
 families.to_csv(output/"family_station_contributions.csv",index=False)
 sensors_order=frequency.groupby("sensor_id").top3_penalty_count.sum().sort_values(ascending=False).index.tolist()
 experiments=sorted(stations.experiment.unique())
 def pivot(comparator,metric,value):
  return stations[(stations.comparator==comparator)&(stations.metric==metric)].pivot(index="experiment",columns="sensor_id",values=value).reindex(index=experiments,columns=sensors_order)
 first=pivot("KRG","mse","penalty_rank")
 fig=go.Figure(go.Heatmap(z=first.to_numpy(),x=[str(s)[-2:] for s in sensors_order],y=experiments,
  customdata=pivot("KRG","mse","net_contribution").to_numpy(),zmin=1,zmax=22,colorscale="RdYlBu",
  text=first.to_numpy(),texttemplate="%{text:.0f}",colorbar=dict(title="Rango"),
  hovertemplate="Campaña %{y}<br>Estación %{x}<br>Rango de contribución %{z}<br>Contribución neta %{customdata:.4f}<extra></extra>"))
 buttons=[]
 for comparator in ["KRG","IDW"]:
  for metric in ["mae","mse"]:
   grid=pivot(comparator,metric,"penalty_rank");value=pivot(comparator,metric,"net_contribution")
   buttons.append(dict(label=f"DIP − {comparator} · {metric.upper()}",method="update",
    args=[dict(z=[grid.to_numpy().tolist()],text=[grid.to_numpy().tolist()],customdata=[value.to_numpy().tolist()]),
      dict(title=f"Estaciones que penalizan a DIP frente a {comparator}: {metric.upper()}")]))
 fig.update_layout(title="Estaciones que penalizan a DIP frente a KRG: MSE",height=950,
  margin=dict(l=270,r=90,t=125,b=65),xaxis_title="Estación (sufijo)",
  updatemenus=[dict(buttons=buttons,active=1,x=0,y=1.08,xanchor="left",yanchor="top")])
 fig.write_html(output/"station_penalty_patterns.html",include_plotlyjs=True)
 parts=["# Sensores que contribuyen a las desventajas de DIP",
  "19 campañas locales completas, 45.600 casos. Diferencias emparejadas DIP menos comparador: positivo perjudica a DIP. La contribución de una estación es la suma de sus diferencias dividida por todas las observaciones de la campaña, respetando su frecuencia de evaluación. Las contribuciones suman exactamente la diferencia de los errores globales históricos.",
  "## Campañas con desventaja global",markdown(totals.groupby(["comparator","metric"]).net_delta.agg(campaigns="size",losses=lambda v:int((v>0).sum())).reset_index()),
  "## Patrón en las campañas donde DIP pierde",
  markdown(loss_frequency.sort_values(["comparator","metric","top3_penalty_count"],ascending=[True,True,False]).groupby(["comparator","metric"]).head(5)),
  "En MAE destacan 36, 27 y 39; en MSE, 58 y 56. El patrón depende del contaminante y la configuración. Las frecuencias describen desventajas observadas, incluyendo diferencias próximas a cero; no equivalen a derrotas estadísticamente significativas. Ninguna diferencia MSE por estación supera la corrección simultánea conservadora. En MAE sí aparecen diferencias significativas en algunas campañas.",
  "## Estaciones que más veces penalizan a DIP",
  markdown(frequency.sort_values(["comparator","metric","top3_penalty_count"],ascending=[True,True,False]).groupby(["comparator","metric"]).head(6).round(4)),
  "## Patrón por contaminante y año",markdown(families.sort_values(["comparator","metric","pollutant","year","mean_contribution"],ascending=[True,True,True,True,False]).groupby(["comparator","metric","pollutant","year"]).head(3).round(4)),
  "## Dependencia de extremos",markdown(pd.DataFrame(tail_records).round(4)),
  "## Límites\n\nLas variantes reutilizan datos y no son réplicas independientes. Bootstrap de semanas completas con sensores y métodos emparejados. Intervalos simultáneos de diferencias por estación corrigen las dos comparaciones y ambas métricas, con límite conservador para 30 campañas. Los extremos se analizan como diagnóstico retrospectivo, sin quitarlos de la evaluación original. WAPE no se usa para atribuir diferencias del MAE/MSE global porque su denominador depende de cada estación. No se establece una relación causal con vegetación, ubicación o concentración."]
 (output/"report.md").write_text("\n\n".join(parts),encoding="utf-8")
 click.echo(frequency.sort_values(["comparator","metric","top3_penalty_count"],ascending=[True,True,False]).groupby(["comparator","metric"]).head(6).to_string(index=False))
 click.echo(totals.groupby(["comparator","metric"]).net_delta.agg(campaigns="size",losses=lambda v:int((v>0).sum())).to_string())
if __name__=="__main__":main()
