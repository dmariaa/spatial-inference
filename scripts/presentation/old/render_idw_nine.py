"""Nine synchronized pollutant IDW cubes, using the shared Madrid grid."""
from pathlib import Path
import json, subprocess
import click
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from metraq_dip.data.aq_backends import get_aq_backend
from metraq_app.data_source import POLLUTANTS
from metraq_app.grid import make_grid, aggregate_cells
from metraq_app.interpolation import reconstruct
from render_idw_month import mapped
from render_idw_presentation import color, font

IDS=(8,7,12,9,10,14,6,30,20)
PALETTE=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]])

def prepare(output,month,reference):
    begin=pd.Timestamp(month+'-01');end=begin+pd.offsets.MonthBegin(1)-pd.Timedelta(hours=1)
    times=pd.date_range(begin-pd.Timedelta(hours=23),end,freq='h')
    manifest=output/'data_manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text(encoding='utf-8'))
        if meta['month']!=month or meta['pollutant_ids']!=list(IDS):
            raise click.ClickException('Cache differs from requested period or pollutants; choose another output directory.')
        return meta,times
    backend=get_aq_backend(dataset='metraq',backend='files')
    data=backend.get_measurements(start_date=times[0].to_pydatetime(),end_date=times[-1].to_pydatetime(),magnitudes=list(IDS))
    data=data.loc[data.is_interpolated.eq(False)&np.isfinite(data.value)].copy()
    sensors=backend.get_sensors(magnitudes=list(IDS))
    grid_sensors=sensors.loc[sensors.id.isin(data.loc[data.magnitude_id.eq(8),'sensor_id'].unique())]
    ctx=make_grid(grid_sensors,1000)
    bounds=np.array([min(p.bounds[0] for p in ctx['grid_cells_m']),min(p.bounds[1] for p in ctx['grid_cells_m']),max(p.bounds[2] for p in ctx['grid_cells_m']),max(p.bounds[3] for p in ctx['grid_cells_m'])])
    with np.load(reference/'data.npz') as ref:
        if not np.array_equal(bounds,ref['bounds']):
            raise click.ClickException('Reference cartography does not match the shared grid.')
    details=[]
    for mid in IDS:
        group=data.loc[data.magnitude_id.eq(mid)]
        hourly={pd.Timestamp(t):g for t,g in group.groupby('entry_date')}
        pred=[];counts=[]
        for t in times:
            if t not in hourly:raise click.ClickException(f'Missing observations for pollutant {mid} at {t}')
            cells,stations=aggregate_cells(ctx,sensors,hourly[t])
            if len(stations)!=hourly[t].sensor_id.nunique():raise click.ClickException(f'Unmapped sensor for {mid} at {t}')
            if len(cells)<2:raise click.ClickException(f'Insufficient spatial coverage for {mid} at {t}')
            surface=reconstruct(ctx['grid'].shape,cells,'idw')
            known=surface[cells.row.to_numpy(int),cells.col.to_numpy(int)]
            if not np.allclose(known,cells.value):raise click.ClickException('Observed values changed')
            if not np.isfinite(surface).all():raise click.ClickException('Nonfinite reconstruction')
            if surface.min()<cells.value.min()-1e-4 or surface.max()>cells.value.max()+1e-4:raise click.ClickException('IDW outside observed range')
            pred.append(surface);counts.append(len(cells))
        pred=np.array(pred);counts=np.array(counts)
        limits=[float(group.value.min()),float(group.value.max())]
        np.savez_compressed(output/f'pollutant_{mid}.npz',idw=pred,timestamps=times.to_numpy(),counts=counts,bounds=bounds,limits=limits)
        name,unit=POLLUTANTS[mid]
        details.append(dict(id=mid,name=name,unit=unit,limits=limits,sensors=int(group.sensor_id.nunique()),min_observed_cells=int(counts.min()),max_observed_cells=int(counts.max()),observed_hours=len(hourly),records=len(group)))
        click.echo(f'Pollutant {mid}: {len(pred)} fields; {counts.min()}-{counts.max()} observed cells/hour')
    meta=dict(month=month,pollutant_ids=list(IDS),pollutants=details,grid_shape=list(ctx['grid'].shape),bounds_utm=bounds.tolist(),metric_crs='EPSG:25830',cell_size_m=1000,window_hours=24,observed_only=True,known_values_preserved=True,full_hourly_coverage=True,idw_power=2,idw_neighbors=8,map_source='OpenStreetMap via Overpass; same reference map, clipped to grid',map_copyright='https://www.openstreetmap.org/copyright')
    manifest.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta,times

@click.command()
@click.option('--month',default='2024-01')
@click.option('--output',type=click.Path(path_type=Path),default='outputs/idw_nine')
@click.option('--reference',type=click.Path(path_type=Path),default='outputs/idw_month')
@click.option('--seconds',type=click.IntRange(10,180),default=62)
@click.option('--fps',type=click.IntRange(1,30),default=24)
@click.option('--preview-only',is_flag=True)
def main(month,output,reference,seconds,fps,preview_only):
    output.mkdir(parents=True,exist_ok=True)
    meta,times=prepare(output,month,reference)
    h,w=meta['grid_shape'];res=384
    mapimage=Image.open(reference/'map.png').convert('RGBA').resize((res,res),Image.Resampling.LANCZOS)
    ink=Image.open(reference/'map_overlay.png').convert('RGBA').resize((res,res),Image.Resampling.LANCZOS)
    map_pixels=np.asarray(ink)
    north=Image.fromarray(np.repeat(map_pixels[:1],res,axis=0))
    east=Image.fromarray(np.repeat(map_pixels[:,-1:],res,axis=1))
    cubes=[]
    for n,detail in enumerate(meta['pollutants']):
        with np.load(output/f"pollutant_{detail['id']}.npz") as data:pred=data['idw']
        low,high=detail['limits'];values=np.clip((pred-low)/max(high-low,1e-6),0,1)*4
        indices=np.minimum(values.astype(int),3);fraction=(values-indices)[...,None]
        colors=np.round(PALETTE[indices]*(1-fraction)+PALETTE[indices+1]*fraction).astype(np.uint8)
        col=n%3;row=n//3;cx=260+col*300;cy=195+row*290
        def p(x,y,z):
            angle=.42;el=.26;world=np.array([(x-.5)*w/h,y-.5,(z-.5)*2.05])
            right=np.array([np.cos(angle),0,np.sin(angle)])
            up=np.array([-np.sin(el)*np.sin(angle),np.cos(el),np.sin(el)*np.cos(angle)])
            toward=np.array([np.cos(el)*np.sin(angle),np.sin(el),-np.cos(el)*np.cos(angle)])
            depth=4-float(toward@world)
            return np.array([cx+600*float(right@world)/depth,cy-600*float(up@world)/depth])
        frontpts=[p(0,1,0),p(1,1,0),p(1,0,0),p(0,0,0)]
        toppts=[p(0,1,0),p(1,1,0),p(1,1,1),p(0,1,1)]
        rightpts=[p(1,1,0),p(1,1,1),p(1,0,1),p(1,0,0)]
        cubes.append((colors,frontpts,toppts,rightpts,detail))
    background=Image.new('RGBA',(1080,1080),'#f8f9fb');d=ImageDraw.Draw(background)
    d.text((540,18),'Madrid · Nueve contaminantes · IDW',anchor='mt',font=font(30),fill='#172b42')
    for _,frontpts,toppts,rightpts,detail in cubes:
        vertices=np.vstack((frontpts,toppts,rightpts))
        center_x=(vertices[:,0].min()+vertices[:,0].max())/2
        bottom_y=vertices[:,1].max()
        d.text((center_x,bottom_y+8),detail['name'],anchor='mt',font=font(25),fill='#172b42')
    d.text((540,1020),'Enero 2024 · Ventana móvil de 24 horas · El tiempo avanza hacia el fondo',anchor='mt',font=font(17),fill='#596a7e')
    d.text((540,1050),'© OpenStreetMap contributors · Rejilla 1 km',anchor='mt',font=font(14),fill='#778595')
    def scene(frameid):
        position=23+(len(times)-24)*frameid/(seconds*fps-1)
        indices=np.clip(position-np.arange(24),0,len(times)-1)
        i0=indices.astype(int);i1=np.minimum(i0+1,len(times)-1);f=(indices-i0)[:,None,None,None]
        im=background.copy()
        for colors,frontpts,toppts,rightpts,detail in cubes:
            window=np.round(colors[i0]*(1-f)+colors[i1]*f).astype(np.uint8)
            front=Image.fromarray(window[0]).resize((res,res),Image.Resampling.BILINEAR).convert('RGBA')
            front=Image.blend(front,mapimage,.16);front.alpha_composite(ink)
            top=Image.fromarray(window[:,0]).resize((res,res),Image.Resampling.BILINEAR).convert('RGBA');top.alpha_composite(north)
            right=Image.fromarray(window[:,:,-1].transpose(1,0,2)).resize((res,res),Image.Resampling.BILINEAR).convert('RGBA');right.alpha_composite(east)
            mapped(im,top,toppts);mapped(im,right,rightpts);mapped(im,front,frontpts)
            ImageDraw.Draw(im).line([tuple(v) for v in frontpts+[frontpts[0]]],fill='#65758a',width=1)
        d=ImageDraw.Draw(im);current=times[int(position)];past=current-pd.Timedelta(hours=23)
        d.text((540,62),f'Frente: {current:%d/%m %H:%M}  |  Fondo: {past:%d/%m %H:%M}',anchor='mt',font=font(19),fill='#596a7e')
        return im.convert('RGB')
    for n in (0,seconds*fps//2,seconds*fps-1):scene(n).save(output/f'preview_{n:04d}.png')
    scene(seconds*fps//2).save(output/'poster.png')
    if preview_only:
        click.echo('Previews ready');return
    path=output/'idw_nueve_contaminantes.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1080x1080','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as enc:
        for frameid in range(seconds*fps):
            enc.stdin.write(scene(frameid).tobytes())
            if frameid%(fps*15)==0:click.echo(f'Rendered {frameid//fps}/{seconds} seconds')
        enc.stdin.close()
        if enc.wait():raise click.ClickException('Encoding failed')
    meta.update(duration_seconds=seconds,fps=fps,resolution=[1080,1080],layout='compact 3x3, centered labels below cubes, no displayed scales',time_direction='Newest at front; history moves backwards',color_limits='Independent per pollutant, fixed over the full month',visual_smoothing='Linear color blend between adjacent hourly reconstructions')
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    click.echo(str(path.resolve()))
if __name__=='__main__':main()
