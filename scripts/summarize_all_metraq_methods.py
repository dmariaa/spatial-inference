"""Compare the validated DIP, Kriging and IDW station ranks in all local campaigns."""
from pathlib import Path
import click
import pandas as pd
import plotly.graph_objects as go

@click.command()
@click.option("--audit",type=click.Path(path_type=Path),default="output/metraq_all_sensor_audit")
def main(audit):
    classical=pd.read_csv(audit/"classical/station_rank_statistics.csv")
    dip=pd.read_csv(audit/"station_rank_statistics.csv").assign(method="DIP")
    merged=pd.concat([dip,classical],ignore_index=True)
    merged.to_csv(audit/"classical/all_methods_rank_statistics.csv",index=False)
    frequency=merged.assign(top3=merged["rank"]<=3,stable_top3=merged.top3_probability>=.95).groupby(["method","metric","sensor_id"]).agg(
        campaigns=("experiment","size"),top3_count=("top3","sum"),
        stable_top3_count=("stable_top3","sum"),min_top3_probability=("top3_probability","min")).reset_index()
    frequency.to_csv(audit/"classical/all_methods_rank_frequency.csv",index=False)
    experiments=sorted(merged.experiment.unique())
    sensors=[28079024,28079058,28079049]+sorted(set(merged.sensor_id)-{28079024,28079058,28079049})
    def grid(method,metric):
        return merged[(merged.method==method)&(merged.metric==metric)].pivot(index="experiment",columns="sensor_id",values="rank").reindex(index=experiments,columns=sensors)
    first=grid("DIP","wape")
    fig=go.Figure(go.Heatmap(z=first.to_numpy(),x=[str(s)[-2:] for s in sensors],y=experiments,
        zmin=1,zmax=22,colorscale="RdYlBu",text=first.to_numpy(),texttemplate="%{text:.0f}",
        hovertemplate="Experimento %{y}<br>Estación %{x}<br>Rango %{z}<extra></extra>",colorbar=dict(title="Rango")))
    buttons=[]
    for method,label in [("DIP","DIP-CNN"),("KRG","Kriging"),("IDW","IDW")]:
        for metric in ["wape","mae","mse"]:
            g=grid(method,metric)
            buttons.append(dict(label=f"{label} · {metric.upper()}",method="update",
                args=[dict(z=[g.to_numpy().tolist()],text=[g.to_numpy().tolist()]),
                      dict(title=f"{label}: rango por {metric.upper()} en las 19 campañas locales")]))
    fig.update_layout(title="DIP-CNN: rango por WAPE en las 19 campañas locales",height=950,
        margin=dict(l=270,r=90,t=120,b=70),xaxis_title="Estación (sufijo)",
        updatemenus=[dict(buttons=buttons,x=0,y=1.08,xanchor="left",yanchor="top")])
    fig.write_html(audit/"classical/all_methods_rank_map.html",include_plotlyjs=True)
    selected=frequency[frequency.sensor_id.isin([28079024,28079058,28079049])]
    click.echo(selected.to_string(index=False))
if __name__=="__main__":main()
