"""Rolling 24-hour IDW volume over a calendar month, with georeferenced OSM context."""
from pathlib import Path
import json, subprocess, urllib.request, urllib.parse
import click
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageChops
from pyproj import Transformer
from metraq_app.data_source import load_history, load_sensors
from metraq_app.grid import make_grid, aggregate_cells
from metraq_app.interpolation import reconstruct
from render_idw_presentation import color, font


def mapped(canvas, tex, pts):
    pts=np.asarray(pts)
    mn=np.floor(pts.min(axis=0)).astype(int);mx=np.ceil(pts.max(axis=0)).astype(int)
    local=pts-mn;tw,th=tex.size
    rows=[];target=[]
    for (x,y),(u,v) in zip(local,[(0,0),(tw,0),(tw,th),(0,th)]):
        rows.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        target.extend([u,v])
    coeff=np.linalg.solve(np.asarray(rows),np.asarray(target))
    warped=tex.transform(tuple(mx-mn),Image.Transform.PERSPECTIVE,tuple(coeff),Image.Resampling.BILINEAR)
    mask=Image.new('L',tuple(mx-mn));ImageDraw.Draw(mask).polygon([tuple(v) for v in local],fill=255)
    warped.putalpha(ImageChops.multiply(warped.getchannel('A'),mask))
    canvas.alpha_composite(warped,tuple(mn))


def basemap(bounds, crs, output):
    cache=output/'osm_features.json'
    x0,y0,x1,y1=bounds
    to_ll=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)
    lon,lat=to_ll.transform([x0,x1,x0,x1],[y0,y0,y1,y1])
    bbox=f'{min(lat)},{min(lon)},{max(lat)},{max(lon)}'
    if not cache.exists():
        query=f'[out:json][timeout:45];(way[highway~"^(motorway|trunk|primary|secondary)$"]({bbox});way[waterway=river]({bbox});way[leisure=park]({bbox});way[landuse=forest]({bbox});way[natural=water]({bbox}););out geom;'
        request=urllib.request.Request('https://overpass-api.de/api/interpreter',data=urllib.parse.urlencode({'data':query}).encode(),headers={'User-Agent':'METRAQ-Research-VideoPrototype/0.3'})
        with urllib.request.urlopen(request,timeout=60) as response:cache.write_bytes(response.read())
    data=json.loads(cache.read_text(encoding='utf-8'))
    if not data.get('elements'):raise click.ClickException('No map features returned')
    size=1000
    image=Image.new('RGBA',(size,size),'#edf0ec');draw=ImageDraw.Draw(image)
    ink=Image.new('RGBA',(size,size));idraw=ImageDraw.Draw(ink)
    project=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
    def coords(geom):
        xs,ys=project.transform([v['lon'] for v in geom],[v['lat'] for v in geom])
        return [((x-x0)/(x1-x0)*size,(y1-y)/(y1-y0)*size) for x,y in zip(xs,ys)]
    elements=data['elements']
    for item in elements:
        tags=item.get('tags',{});pts=coords(item.get('geometry',[]))
        if len(pts)<3 or 'highway' in tags or 'waterway' in tags:continue
        if item.get('nodes',[0])[0]!=item.get('nodes',[1])[-1]:continue
        water=tags.get('natural')=='water'
        draw.polygon(pts,fill='#b5d7e7' if water else '#c4d5bc')
        idraw.polygon(pts,fill=(95,160,200,95) if water else (65,110,55,55))
    for item in elements:
        tags=item.get('tags',{});pts=coords(item.get('geometry',[]))
        if len(pts)<2:continue
        if 'waterway' in tags:
            draw.line(pts,fill='#73a9c5',width=5);idraw.line(pts,fill=(40,110,160,230),width=4)
        elif 'highway' in tags:
            width=4 if tags['highway'] in ('motorway','trunk') else 2
            draw.line(pts,fill='#ffffff',width=width+2);draw.line(pts,fill='#9f9e97',width=width)
            idraw.line(pts,fill=(245,246,243,185),width=width+2);idraw.line(pts,fill=(45,55,65,155),width=max(1,width-1))
    # Geographic labels positioned in the same projected coordinate system.
    for name,lon,lat in [('Madrid',-3.7038,40.4168),('El Retiro',-3.6815,40.4153),('Casa de Campo',-3.756,40.419),('Manzanares',-3.722,40.404)]:
        xx,yy=coords([{'lon':lon,'lat':lat}])[0]
        draw.text((xx,yy),name,font=font(24),fill='#293d46',stroke_width=2,stroke_fill='#f6f7f4')
        idraw.text((xx,yy),name,font=font(22),fill=(20,30,40,255),stroke_width=2,stroke_fill=(248,249,250,210))
    image.save(output/'map.png');ink.save(output/'map_overlay.png')
    return image,ink,len(elements)

@click.command()
@click.option('--month',default='2024-01')
@click.option('--output',type=click.Path(path_type=Path),default='outputs/idw_month')
@click.option('--seconds',type=click.IntRange(10,180),default=62)
@click.option('--fps',type=click.IntRange(1,30),default=24)
def main(month,output,seconds,fps):
    output.mkdir(parents=True,exist_ok=True)
    begin=pd.Timestamp(month+'-01');end=begin+pd.offsets.MonthBegin(1)-pd.Timedelta(hours=1)
    times=pd.date_range(begin-pd.Timedelta(hours=23),end,freq='h')
    cache=output/'data.npz'
    if cache.exists():
        with np.load(cache) as saved:
            if not np.array_equal(saved['timestamps'],times.to_numpy()):raise click.ClickException('Cached period mismatch; choose a new output directory')
            predictions=saved['idw'];bounds=saved['bounds'];counts=saved['counts'];low,high=saved['limits']
    else:
        click.echo('Loading real hourly observations...')
        measurements=load_history('files',8,times[0].to_pydatetime(),times[-1].to_pydatetime())
        measurements=measurements.loc[measurements.is_interpolated.eq(False)&np.isfinite(measurements.value)].copy()
        sensors=load_sensors('files',(8,));sensors=sensors.loc[sensors.id.isin(measurements.sensor_id.unique())]
        ctx=make_grid(sensors,1000)
        bounds=np.array([min(p.bounds[0] for p in ctx['grid_cells_m']),min(p.bounds[1] for p in ctx['grid_cells_m']),max(p.bounds[2] for p in ctx['grid_cells_m']),max(p.bounds[3] for p in ctx['grid_cells_m'])])
        groups={pd.Timestamp(t):g for t,g in measurements.groupby('entry_date')}
        predictions=[];counts=[]
        for t in times:
            if t not in groups:raise click.ClickException(f'Missing observations: {t}')
            cells,_=aggregate_cells(ctx,sensors,groups[t]);counts.append(len(cells))
            if len(cells)<2:raise click.ClickException(f'Insufficient cells: {t}')
            predictions.append(reconstruct(ctx['grid'].shape,cells,'idw'))
        predictions=np.array(predictions);counts=np.array(counts)
        if not np.isfinite(predictions).all():raise click.ClickException('Non-finite IDW fields')
        low=float(measurements.value.min());high=float(measurements.value.max())
        np.savez_compressed(cache,idw=predictions,timestamps=times.to_numpy(),bounds=bounds,counts=counts,limits=[low,high])
        click.echo(f'{len(times)} hourly fields prepared; {counts.min()}–{counts.max()} observed cells/hour')
    mapimage,ink,nfeatures=basemap(bounds,'EPSG:25830',output)
    h,w=predictions.shape[1:]
    # Grid row zero is NORTH. Preserve that orientation on the front map.
    anchors=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]])
    values=np.clip((predictions-low)/(high-low),0,1)*4
    indices=np.minimum(values.astype(int),3);fraction=(values-indices)[...,None]
    colors=np.round(anchors[indices]*(1-fraction)+anchors[indices+1]*fraction).astype(np.uint8)
    def p(x,y,z):
        angle=.42;el=.26
        world=np.array([(x-.5)*w/h,y-.5,(z-.5)*2.05])
        right=np.array([np.cos(angle),0,np.sin(angle)])
        up=np.array([-np.sin(el)*np.sin(angle),np.cos(el),np.sin(el)*np.cos(angle)])
        toward_camera=np.array([np.cos(el)*np.sin(angle),np.sin(el),-np.cos(el)*np.cos(angle)])
        depth=4-float(toward_camera@world)
        return np.array([485+970*float(right@world)/depth,285-970*float(up@world)/depth])
    frontpts=[p(0,1,0),p(1,1,0),p(1,0,0),p(0,0,0)]
    # Extend the cartographic intersections at the north/east borders along time.
    # Each line remains at its true spatial position for the complete 24-hour depth.
    map_pixels=np.asarray(ink)
    north=np.repeat(map_pixels[0:1,:,:],1000,axis=0)
    east=np.repeat(map_pixels[:,-1:,:],1000,axis=1)
    north_overlay=Image.fromarray(north).convert('RGBA')
    east_overlay=Image.fromarray(east).convert('RGBA')
    inset=mapimage.convert('RGB').resize((290,290),Image.Resampling.LANCZOS)
    for frameid in range(seconds*fps):
        position=23+(len(times)-24)*frameid/(seconds*fps-1)
        ages=np.arange(24)
        indices=np.clip(position-ages,0,len(times)-1)
        i0=indices.astype(int);i1=np.minimum(i0+1,len(times)-1);f=(indices-i0)[:,None,None,None]
        window=np.round(colors[i0]*(1-f)+colors[i1]*f).astype(np.uint8)
        front=Image.fromarray(window[0]).resize((1000,1000),Image.Resampling.BILINEAR).convert('RGBA')
        front=Image.blend(front,mapimage,.22);front.alpha_composite(ink)
        # At the top spatial edge, row zero is north; time runs from newest to oldest.
        top=Image.fromarray(window[:,0]).resize((1000,1000),Image.Resampling.BILINEAR).convert('RGBA');top.alpha_composite(north_overlay)
        right=Image.fromarray(window[:,:,-1].transpose(1,0,2)).resize((1000,1000),Image.Resampling.BILINEAR).convert('RGBA');right.alpha_composite(east_overlay)
        im=Image.new('RGBA',(1280,720),'#f8f9fb')
        mapped(im,top,[p(0,1,0),p(1,1,0),p(1,1,1),p(0,1,1)])
        mapped(im,right,[p(1,1,0),p(1,1,1),p(1,0,1),p(1,0,0)])
        mapped(im,front,frontpts)
        d=ImageDraw.Draw(im)
        d.line([tuple(v) for v in frontpts+[frontpts[0]]],fill='#48586b',width=1)
        d.text((55,35),'Madrid · Reconstrucción IDW de NO₂',font=font(33),fill='#172b42')
        d.text((55,84),f'{begin:%d/%m/%Y} – {end:%d/%m/%Y} · ventana móvil de 24 horas',font=font(22),fill='#596a7e')
        current=times[int(position)];past=current-pd.Timedelta(hours=23)
        d.text((855,155),'Zona de estudio',font=font(23),fill='#172b42')
        im.alpha_composite(inset.convert('RGBA'),(855,195))
        d.text((855,498),'N ↑ · vías principales, parques y río',font=font(15),fill='#596a7e')
        d.text((855,535),'© OpenStreetMap contributors',font=font(15),fill='#596a7e')
        a=p(1,0,0)+[8,25];b=p(1,0,1)+[8,25]
        d.line([tuple(a),tuple(b)],fill='#62758b',width=2)
        v=(b-a)/np.linalg.norm(b-a);orth=np.array([-v[1],v[0]])
        d.polygon([tuple(b),tuple(b-v*12+orth*5),tuple(b-v*12-orth*5)],fill='#62758b')
        for age in (0,6,12,18,23):
            q=a+(b-a)*age/23
            d.text(tuple(q+[0,10]),'Ahora' if age==0 else f'−{age} h',font=font(15),fill='#596a7e')
        d.text(tuple(p(0,1,0)+[-28,-32]),'N',font=font(18),fill='#172b42')
        d.text((55,595),f'Cara frontal: {current:%d/%m/%Y %H:%M}   |   Fondo: {past:%d/%m/%Y %H:%M}',font=font(23),fill='#172b42')
        d.text((55,637),'Las horas nuevas entran delante; la historia se desplaza hacia el fondo.',font=font(18),fill='#596a7e')
        for x in range(230):d.line([(900+x,610),(900+x,626)],fill=color(low+(high-low)*x/229,low,high))
        d.text((900,643),f'{low:.0f}',font=font(15),fill='#596a7e');d.text((1040,643),f'{high:.0f} µg/m³',font=font(15),fill='#596a7e')
        d.text((55,686),'IDW · rejilla 1 km · escala fija durante todo el mes · animación suavizada entre horas',font=font(15),fill='#778595')
        if frameid in (0,seconds*fps//2,seconds*fps-1):im.convert('RGB').save(output/f'preview_{frameid:04d}.png')
        if frameid==0:
            path=output/'idw_mes_ventana24h.mp4'
            cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
            enc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        enc.stdin.write(im.convert('RGB').tobytes())
    enc.stdin.close()
    if enc.wait():raise click.ClickException('Encoding failed')
    meta={'month':month,'pollutant':'NO2','units':'µg/m³','window_hours':24,'latest_at_front':True,'older_to_back':True,'volume_depth':2.05,'map_extrusion':'North and east boundary intersections extended along the entire time axis','camera_projection':'pinhole perspective, yaw 24 degrees, elevation 15 degrees','grid_rows':'north to south','grid_shape':[h,w],'bounds_utm':bounds.tolist(),'metric_crs':'EPSG:25830','color_limits':[float(low),float(high)],'observed_cells_min':int(counts.min()),'observed_cells_max':int(counts.max()),'hourly_fields':len(times),'map_features':nfeatures,'map_source':'OpenStreetMap via Overpass API; contemporary map, not historical 2024','map_copyright':'https://www.openstreetmap.org/copyright','visual_smoothing':'Linear blend of hourly colors; IDW computed independently each hour','duration_seconds':seconds,'fps':fps}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    click.echo(str(path.resolve()))
if __name__=='__main__':main()
