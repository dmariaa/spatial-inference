"""Textured space-time volumes, with moving camera and temporal section."""
from pathlib import Path
import json
import subprocess
import click
import numpy as np
from PIL import Image, ImageDraw
from render_idw_presentation import color, font

@click.command()
@click.option('--source',type=click.Path(path_type=Path),default='outputs/idw_presentation')
@click.option('--output',type=click.Path(path_type=Path),default='outputs/idw_volume_v2')
@click.option('--fps',type=click.IntRange(1,30),default=24)
def main(source,output,fps):
    output.mkdir(parents=True,exist_ok=True)
    with np.load(source/'data.npz') as data:
        obs=data['observations']; pred=data['idw']; times=data['timestamps']
    meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    low,high=meta['color_limits']; nt,h,w=obs.shape
    colors=np.array([color(v,low,high) for v in pred.ravel()],dtype=np.uint8).reshape(*pred.shape,3)
    observed_colors=np.zeros((*obs.shape,4),dtype=np.uint8)
    for t,y,x in np.argwhere(np.isfinite(obs)):
        observed_colors[t,y,x]=(*color(obs[t,y,x],low,high),255)
    def texture(a):return Image.fromarray(np.ascontiguousarray(a),'RGBA' if a.shape[-1]==4 else 'RGB').convert('RGBA')
    front=[texture(colors[t,::-1]) for t in range(nt)]
    top=texture(colors[:,-1].transpose(0,1,2))
    # Texture horizontal coordinate is time, vertical coordinate is northing.
    right=texture(colors[:,:, -1].transpose(1,0,2)[::-1])
    def smooth(p):p=np.clip(p,0,1);return p*p*(3-2*p)
    def frame(sec):
        im=Image.new('RGBA',(1280,720),(248,249,251,255))
        d=ImageDraw.Draw(im)
        blend=smooth((sec-10)/3)
        cut=smooth((sec-17)/7)*.78
        angle=.60+.22*np.sin(sec/28*np.pi*2)
        elevation=.24+.05*np.sin(sec/28*np.pi)
        scale=245+18*np.sin(sec/28*np.pi)
        def p(x,y,z):
            x-=.5;y-=.5;z=(z-.5)*1.7
            return np.array([600+scale*(np.cos(angle)*x+np.sin(angle)*z),335+scale*(np.sin(elevation)*np.sin(angle)*x-np.cos(elevation)*y-np.sin(elevation)*np.cos(angle)*z)])
        def quad(points):return [tuple(v) for v in points]
        def mapped(canvas,tex,points):
            # Affine texture projection onto a planar parallelogram.
            a,b,_,c=np.array(points); basis=np.column_stack((b-a,c-a)); inv=np.linalg.inv(basis)
            mn=np.floor(np.min(points,axis=0)).astype(int);mx=np.ceil(np.max(points,axis=0)).astype(int)
            size=tuple((mx-mn).astype(int));origin=inv@(mn-a)
            tw,th=tex.size
            coeff=(inv[0,0]*tw,inv[0,1]*tw,origin[0]*tw,inv[1,0]*th,inv[1,1]*th,origin[1]*th)
            warped=tex.transform(size,Image.Transform.AFFINE,coeff,Image.Resampling.BILINEAR)
            canvas.alpha_composite(warped,tuple(mn))
        def outline(canvas):
            draw=ImageDraw.Draw(canvas)
            for a,b in [((0,0,0),(1,0,0)),((1,0,0),(1,1,0)),((1,1,0),(0,1,0)),((0,1,0),(0,0,0)),((1,0,0),(1,0,1)),((1,1,0),(1,1,1)),((0,1,0),(0,1,1)),((1,0,1),(1,1,1)),((1,1,1),(0,1,1))]:
                draw.line([tuple(p(*a)),tuple(p(*b))],fill=(130,140,151,160),width=1)
        observed=Image.new('RGBA',im.size)
        outline(observed); od=ImageDraw.Draw(observed)
        voxels=[]
        for t,y,x in np.argwhere(np.isfinite(obs)):
            z0=t/nt;z1=(t+1)/nt;x0=x/w;x1=(x+1)/w;y0=y/h;y1=(y+1)/h
            c=tuple(observed_colors[t,y,x,:3])
            faces=[([p(x0,y0,z0),p(x1,y0,z0),p(x1,y1,z0),p(x0,y1,z0)],c),([p(x1,y0,z0),p(x1,y0,z1),p(x1,y1,z1),p(x1,y1,z0)],tuple(int(v*.83) for v in c)),([p(x0,y1,z0),p(x1,y1,z0),p(x1,y1,z1),p(x0,y1,z1)],tuple(int(v*.94) for v in c))]
            depth=np.sin(angle)*(x0+.5/w)+np.sin(elevation)*y0-np.cos(angle)*z0*1.7
            voxels.append((depth,faces))
        for _,faces in sorted(voxels,key=lambda v:v[0]):
            for pts,c in faces:od.polygon(quad(pts),fill=(*c,255))
        volume=Image.new('RGBA',im.size)
        # Crop the exposed outer surfaces to the remaining time interval.
        k=min(nt-1,int(cut*nt)); actual=k/nt
        mapped(volume,top.crop((0,k,w,nt)),[p(0,1,actual),p(1,1,actual),p(1,1,1),p(0,1,1)])
        mapped(volume,right.crop((k,0,nt,h)),[p(1,1,actual),p(1,1,1),p(1,0,1),p(1,0,actual)])
        mapped(volume,front[k],[p(0,1,actual),p(1,1,actual),p(1,0,actual),p(0,0,actual)])
        vd=ImageDraw.Draw(volume)
        vd.line(quad([p(0,1,actual),p(1,1,actual),p(1,0,actual),p(0,0,actual),p(0,1,actual)]),fill='#586878',width=1)
        result=Image.blend(observed,volume,float(blend))
        im.alpha_composite(result)
        d=ImageDraw.Draw(im)
        title='Observaciones' if sec<10 else 'De las observaciones a IDW' if sec<13 else 'Reconstrucción IDW'
        d.text((65,45),title,font=font(35),fill='#172b42')
        d.text((65,94),'Madrid · NO₂ · 15 enero 2024 · 24 horas',font=font(22),fill='#596a7e')
        # The time axis recedes into the volume, matching the reference.
        a=p(1,0,0)+[14,30];b=p(1,0,1)+[14,30]
        d.line([tuple(a),tuple(b)],fill='#62758b',width=2)
        for hour in (0,6,12,18,23):
            q=a+(b-a)*hour/23
            d.line([tuple(q),tuple(q+[0,6])],fill='#62758b',width=1)
            d.text(tuple(q+[0,10]),f'{hour:02d} h',font=font(14),fill='#596a7e')
        d.text(tuple(b+[22,-12]),'Tiempo',font=font(18),fill='#172b42')
        d.text((70,590),'El mapa ocupa la cara frontal; el tiempo se extiende hacia el fondo.',font=font(21),fill='#172b42')
        subtitle='Celdas observadas; el espacio sin mediciones queda vacío.' if sec<10 else 'IDW completa el espacio entre sensores y conserva las observaciones.' if sec<17 else f'Corte temporal en movimiento · cara visible: {k:02d}:00'
        d.text((70,628),subtitle,font=font(19),fill='#596a7e')
        for x in range(250):d.line([(925+x,628),(925+x,642)],fill=color(low+(high-low)*x/249,low,high))
        d.text((925,650),f'{low:.0f}',font=font(15),fill='#596a7e');d.text((1090,650),f'{high:.0f} µg/m³',font=font(15),fill='#596a7e')
        d.text((70,685),'PROTOTIPO 02 · volumen espaciotemporal · rejilla 1 km · sin audio',font=font(14),fill='#778595')
        return im.convert('RGB')
    path=output/'observaciones_idw_volumen.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as enc:
        for i in range(28*fps):
            im=frame(i/fps)
            if i in (5*fps,15*fps,22*fps):im.save(output/f'preview_{i//fps:02d}.png')
            enc.stdin.write(im.tobytes())
        enc.stdin.close()
        if enc.wait():raise click.ClickException('Encoding failed')
    meta.update(duration_seconds=28,fps=fps,rendering='Textured space-time block; sparse observed voxels; moving camera; temporal cut',time_axis='depth')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    click.echo(str(path.resolve()))
if __name__=='__main__':main()
