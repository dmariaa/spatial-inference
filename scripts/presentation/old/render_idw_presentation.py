"""Render a reproducible METRAQ observation-to-IDW presentation prototype."""
from pathlib import Path
import json
import subprocess
import click
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from metraq_app.data_source import load_history, load_sensors
from metraq_app.grid import make_grid, aggregate_cells
from metraq_app.interpolation import reconstruct

FONT = 'C:/Windows/Fonts/segoeui.ttf'
def font(size):
    return ImageFont.truetype(FONT, size)

def color(value, low, high):
    anchors = np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]])
    p = np.clip((value-low)/(high-low),0,1)*4
    i = min(int(p),3)
    return tuple(np.round(anchors[i]*(1-(p-i))+anchors[i+1]*(p-i)).astype(int))

@click.command()
@click.option('--start', default='2024-01-15')
@click.option('--output', type=click.Path(path_type=Path), default='outputs/idw_presentation')
@click.option('--fps', default=12, type=click.IntRange(1,30))
def main(start, output, fps):
    output.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start, periods=24, freq='h')
    measurements = load_history('files',8,times[0].to_pydatetime(),times[-1].to_pydatetime())
    measurements = measurements.loc[measurements.is_interpolated.eq(False)].copy()
    if measurements.empty:
        raise click.ClickException('No observed NO2 records in the requested period.')
    sensors = load_sensors('files',(8,))
    sensors = sensors.loc[sensors.id.isin(measurements.sensor_id.unique())].copy()
    ctx = make_grid(sensors,1000)
    obs, predictions, counts = [], [], []
    for t in times:
        cells,_ = aggregate_cells(ctx,sensors,measurements.loc[measurements.entry_date.eq(t)])
        if len(cells)<2:
            raise click.ClickException(f'Insufficient observations at {t}.')
        obs.append(reconstruct(ctx['grid'].shape,cells,'observed'))
        predictions.append(reconstruct(ctx['grid'].shape,cells,'idw'))
        counts.append(len(cells))
    obs, predictions = np.array(obs),np.array(predictions)
    assert np.isfinite(predictions).all()
    mask = np.isfinite(obs)
    assert np.allclose(obs[mask],predictions[mask])
    low, high = float(np.nanmin(obs)),float(np.nanmax(obs))
    high=max(high,low+1)
    h,w=obs.shape[1:]
    def scene(seconds):
        im=Image.new('RGB',(1280,720),'#101b2b'); d=ImageDraw.Draw(im)
        phase=0 if seconds<10 else 1 if seconds<20 else 2
        d.text((56,34),'DEL SENSOR AL CAMPO ESPACIAL',font=font(30),fill='#f3f7fc')
        d.text((56,82),f'Madrid | NO₂ | {times[0]:%d/%m/%Y} | 24 horas',font=font(22),fill='#a9c0d8')
        titles=['01  Observaciones','02  Reconstrucción IDW','03  Mismo espacio, mismo tiempo']
        d.text((56,132),titles[phase],font=font(27),fill='#65dfd1')
        index=min(23,int((seconds%10)/10*24))
        def cube(data,cx,cy,size,layers):
            def xy(x,y,z):
                return (cx+size*(x/w-y/h),cy+size*.36*(x/w+y/h)-z*9)
            for z in range(layers):
                a=data[z]
                for y in range(h-1,-1,-1):
                    for x in range(w):
                        if np.isfinite(a[y,x]):
                            pts=[xy(x,y,z),xy(x+1,y,z),xy(x+1,y+1,z),xy(x,y+1,z)]
                            d.polygon(pts,fill=color(a[y,x],low,high))
                corners=[xy(0,0,z),xy(w,0,z),xy(w,h,z),xy(0,h,z)]
                d.line(corners+[corners[0]],fill='#51677e',width=1)
            base=xy(0,0,0); top=xy(0,0,23)
            d.line([base,top],fill='#e1eaf5',width=2)
            d.text((top[0]-65,top[1]-28),'Tiempo ↑',font=font(16),fill='#d2e1ef')
            d.text((cx-size,cy+size*.36+18),'Oeste',font=font(15),fill='#a9c0d8')
            d.text((cx+size-48,cy+size*.36+18),'Este',font=font(15),fill='#a9c0d8')
            return xy
        if phase<2:
            cube(obs if phase==0 else predictions,480,440,260,index+1)
            d.text((860,240),f'{times[index]:%H:%M}',font=font(48),fill='white')
            text=['Solo celdas observadas','Huecos = ausencia de datos'] if phase==0 else ['Estimación entre sensores','IDW · potencia 2 · 8 vecinos']
            for j,line in enumerate(text):d.text((850,320+j*38),line,font=font(20),fill='#d2e1ef')
            d.text((850,415),f'{counts[index]} celdas observadas',font=font(19),fill='#a9c0d8')
        else:
            cube(obs,335,470,205,24);cube(predictions,925,470,205,24)
            d.text((215,595),'OBSERVACIONES',font=font(22),fill='white')
            d.text((795,595),'RECONSTRUCCIÓN IDW',font=font(22),fill='white')
        for x in range(360):d.line([(460+x,660),(460+x,675)],fill=color(low+(high-low)*x/359,low,high))
        d.text((395,651),f'{low:.0f}',font=font(17),fill='white')
        d.text((835,651),f'{high:.0f} µg/m³',font=font(17),fill='white')
        d.text((56,686),'PROTOTIPO · rejilla de 1 km · cada capa representa una hora · sin audio',font=font(15),fill='#a9c0d8')
        return im
    path=output/'observaciones_idw.mp4'
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    cache = {}
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        for i in range(30*fps):
            seconds = i/fps
            phase = min(2,int(seconds//10))
            key = (phase, min(23,int((seconds%10)/10*24)) if phase<2 else 0)
            if key not in cache: cache[key] = scene(seconds)
            frame = cache[key]
            if i in (5*fps,15*fps,25*fps):frame.save(output/f'preview_{i//fps:02d}.png')
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait()!=0:raise click.ClickException('Video encoding failed.')
    np.savez_compressed(output/'data.npz',observations=obs,idw=predictions,timestamps=times.to_numpy())
    metadata={'pollutant':'NO2','units':'µg/m³','start':str(times[0]),'end':str(times[-1]),'records':len(measurements),'sensors':len(sensors),'observed_cells_per_hour':counts,'grid_shape':[h,w],'cell_size_m':1000,'color_limits':[low,high],'observed_only':True,'idw_power':2,'idw_neighbors':8,'duration_seconds':30,'fps':fps,'resolution':[1280,720],'purpose':'Visual presentation prototype; no held-out accuracy evaluation.'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    click.echo(json.dumps(metadata,ensure_ascii=False))
    click.echo(str(path.resolve()))

if __name__=='__main__':main()

