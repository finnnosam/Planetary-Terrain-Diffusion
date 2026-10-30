"""Design illustrations from local Earth data and an explicitly analytic schematic.

No model inference/training, no selected checkpoint edits. Semantic schematic
fields are authored examples, not an implemented landform-label extractor.
"""
import json
import html
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

from planet_training.terrain_structure import conservative_resample
from scripts.diagnose_terrain_structure import sample,image,legend,gauss

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/terrain-flow-plan'
CASES=[('alps','Alpine arc',45.5,9.),('california','California: ranges / valley',37.,-120.),
       ('plains','Rockies / Great Plains transition',40.,-100.),('plateau','High plateau context',34.,88.)]
EXTENT=1290.24


def resample(z,n,spacing=2880.):
    return conservative_resample(torch.from_numpy(z)[None,None],n,spacing).numpy()[0,0]


def axis(ax,field,title,spacing=2880.,categorical=False):
    if categorical:ax.imshow(field,cmap=ListedColormap(['#306c94','#a3b58a']),vmin=0,vmax=1,extent=(-EXTENT/2,EXTENT/2,-EXTENT/2,EXTENT/2),interpolation='nearest')
    else:ax.imshow(image(field,spacing),extent=(-EXTENT/2,EXTENT/2,-EXTENT/2,EXTENT/2),interpolation='nearest')
    ax.set_title(title,fontsize=9);ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=7)


def land_moments(z,mask,sigma):
    den=gauss(mask.astype(float),sigma).clip(1e-8)
    mean=gauss(z*mask,sigma)/den
    relief=np.sqrt(np.maximum(gauss(z*z*mask,sigma)/den-mean*mean,0))
    return mean,relief*mask


def schematic(out):
    q=(np.arange(448)+.5-224)*2.88;xx,yy=np.meshgrid(q,-q)
    # Invented layout, not a fit to Earth: coast / secondary belt / lowland /
    # primary belt / elevated quiet plateau. Local domain is not a whole planet.
    coast=-410+35*np.sin(yy/180)
    distance=xx-coast;land=distance>0
    primary_axis=95+55*np.sin(yy/235);secondary_axis=-220+30*np.sin(yy/185)
    primary=np.exp(-.5*((xx-primary_axis)/42)**2)
    secondary=np.exp(-.5*((xx-secondary_axis)/24)**2)
    low=np.exp(-.5*((xx+70-20*np.sin(yy/210))/62)**2)
    plateau=1100/(1+np.exp(-(xx-310)/30))
    floor=160+70*(yy+EXTENT/2)/EXTENT
    structural=floor+primary*(2400+300*np.cos(yy/230))+secondary*(850+100*np.cos(yy/140))+plateau
    taper=np.clip(distance/100,0,1)
    land_height=structural*taper
    ocean=-5300*(1-np.exp(np.minimum(distance,0)/130))
    z=np.where(land,land_height,ocean)
    mountain=(primary+.55*secondary).clip(0,1)*land
    lower=low*land
    # Conceptual detailed terrain: aligned modulations, not a production generator.
    texture=(np.sin((xx-primary_axis)/7+.6*np.sin(yy/18))*np.cos(yy/23)
             +.4*np.sin(xx/3.5+yy/8))
    fine=z+land*taper*(130*primary+60*secondary+4)*texture
    coarse=resample(z.astype(np.float32),8);z40=resample(z.astype(np.float32),32)
    mask40=z40>0
    _,local=land_moments(z40,mask40,80.64/40.32)
    _,broad=land_moments(z40,mask40,240./40.32)
    broad8=resample(broad.astype(np.float32),8,40320.)
    mountain40=resample(mountain.astype(np.float32),32)*mask40
    lower40=resample(lower.astype(np.float32),32)*mask40
    # Signed distance to the ideal analytic shoreline, calculated geometrically.
    # This schematic's coast is known analytically; production needs spherical
    # distance from the actual sampled coastline, not this planar approximation.
    signed=distance/np.sqrt(1+(35/180*np.cos(yy/180))**2)
    gy,gx=np.gradient(fine,2.88);slope=np.hypot(gx,gy)
    fig,axs=plt.subplots(4,4,figsize=(14,13.8),constrained_layout=True)
    axis(axs[0,0],land,'1 · Land/ocean\nIllustrative local coastline',categorical=True)
    im=axs[0,1].imshow(signed,cmap='RdBu_r',vmin=-400,vmax=400,extent=(-EXTENT/2,EXTENT/2,-EXTENT/2,EXTENT/2));axs[0,1].set_title('Derived · Signed coast distance\nkm; positive inland',fontsize=9)
    fig.colorbar(im,ax=axs[0,1],orientation='horizontal',fraction=.035,pad=.1).set_label('km (display clipped)',fontsize=8)
    axs[0,2].imshow(land,cmap=ListedColormap(['#e0e5e8','#eef0e9']),extent=(-EXTENT/2,EXTENT/2,-EXTENT/2,EXTENT/2))
    sel=np.zeros_like(land);ys=np.arange(0,448,40);sel[ys,np.abs(distance[ys]).argmin(1)]=True
    dx=np.ones_like(xx);dy=-(35/180)*np.cos(yy/180);length=np.hypot(dx,dy)
    axs[0,2].quiver(xx[sel],yy[sel],60*dx[sel]/length[sel],60*dy[sel]/length[sel],angles='xy',scale_units='xy',scale=1,color='#34434a')
    axs[0,2].set_title('Derived · Coast normals\nDirection, not a learned extra head',fontsize=9)
    axs[0,3].text(.5,.5,'Global input: requested land fraction\n\nThis is one regional crop.\nIts local land fraction is not\nthe global requested percentage.',ha='center',va='center',transform=axs[0,3].transAxes,fontsize=10);axs[0,3].set_axis_off()
    axis(axs[1,0],coarse,'2 · Broad elevation / bathymetry\n161.28 km cell means',161280.)
    im=axs[1,1].imshow(broad8,cmap='magma',vmin=0,vmax=1000,extent=(-EXTENT/2,EXTENT/2,-EXTENT/2,EXTENT/2),interpolation='nearest');axs[1,1].set_title('Proposed learned · Broad relief potential\nIllustrated from structural truth, not coarse height',fontsize=9)
    fig.colorbar(im,ax=axs[1,1],orientation='horizontal',fraction=.035,pad=.1).set_label('Regional RMS relief, m (display clipped)',fontsize=8)
    axis(axs[1,2],z40,'Context · Desired structural elevation\n40.32 km, supplied analytic reference',40320.)
    axs[1,3].text(.5,.5,'Relief potential is a soft guide:\n\nIt leaves room for a range,\nan adjacent low region,\nand quiet high plateaus.\n\nIt is not a hard elevation floor\nor a mountain-coverage quota.',ha='center',va='center',transform=axs[1,3].transAxes,fontsize=10);axs[1,3].set_axis_off()
    axis(axs[2,0],z40,'3 · Structural elevation\nJointly generated with the next three fields',40320.)
    for ax,field,title in [(axs[2,1],mountain40,'Mountain-system support\nPrimary and secondary belt extent'),(axs[2,2],lower40,'Lower-region support\nConnected corridor between ranges')]:
        im=ax.imshow(field,cmap='viridis',vmin=0,vmax=1,extent=(-EXTENT/2,EXTENT/2,-EXTENT/2,EXTENT/2),interpolation='nearest');ax.set_title(title,fontsize=9)
        fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.035,pad=.1).set_label('Illustrative support 0–1; not calibrated probability',fontsize=8)
    im=axs[2,3].imshow(local,cmap='magma',vmin=0,vmax=1000,extent=(-EXTENT/2,EXTENT/2,-EXTENT/2,EXTENT/2),interpolation='nearest');axs[2,3].set_title('Local relief · sigma 80.64 km\nQuiet elevated plateau at right',fontsize=9)
    fig.colorbar(im,ax=axs[2,3],orientation='horizontal',fraction=.035,pad=.1).set_label('RMS relief, m (display clipped)',fontsize=8)
    axis(axs[3,0],fine,'4 · Detailed elevation\nAnalytic detail illustration, 2.88 km',2880.)
    im=axs[3,1].imshow(slope,cmap='cividis',vmin=0,vmax=70,extent=(-EXTENT/2,EXTENT/2,-EXTENT/2,EXTENT/2));axs[3,1].set_title('Derived final field · Slope\nNo separate generator required',fontsize=9)
    fig.colorbar(im,ax=axs[3,1],orientation='vertical',fraction=.035,pad=.03,ticks=[0,35,70]).set_label('m/km (display clipped)',fontsize=8)
    axs[3,2].text(.5,.5,'Optional later: drainage\n\nDerive a global drainage graph\nfrom the final elevation.\nLabel real basins and outlets.\nDo not call local slope vectors\na globally valid river network.',ha='center',va='center',transform=axs[3,2].transAxes,fontsize=10);axs[3,2].set_axis_off()
    axs[3,3].text(.5,.5,'All four rows: same 1,290.24 km\n\nElevation colours: ±6,000 m\nSea level: 0 m\nNW light, 5× slope shading\n\nConstructed design illustration.\nNot Earth, not a trained result.',ha='center',va='center',transform=axs[3,3].transAxes,fontsize=10);axs[3,3].set_axis_off()
    for ax in axs.flat:
        if ax.axison:ax.set_xlabel('km');ax.set_ylabel('km');ax.tick_params(labelsize=7)
    fig.suptitle('Ideal field relationships — analytic schematic, not a generation result\nOne invented region: coast → secondary range → connected lower corridor → primary range → quiet high plateau',fontsize=12)
    legend(fig,axs)
    fig.savefig(out/'ideal-field-relationships.png',dpi=125);plt.close(fig)
    np.savez_compressed(out/'ideal-schematic.npz',elevation_fine_metres=fine.astype(np.float32),structural40_metres=z40,
        broad161_metres=coarse,mountain_support=mountain40.astype(np.float32),lower_support=lower40.astype(np.float32),
        local_relief40_metres=local.astype(np.float32),regional_relief40_metres=broad.astype(np.float32),coast_distance_km=signed.astype(np.float32))


def report(out):
    source=ROOT/'docs/terrain-generation-flow-plan.md';text=source.read_text(encoding='utf-8')
    def slug(value):return re.sub(r'[^a-z0-9]+','-',value.lower()).strip('-')
    def inline(value):
        value=html.escape(value)
        def link(match):
            label,path=match.groups();target=(source.parent/html.unescape(path)).resolve()
            relative=os.path.relpath(target,out).replace('\\','/')
            return '<a href="'+html.escape(relative,quote=True)+'">'+label+'</a>'
        value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,value)
        value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
        return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)
    def blocks(lines):
        output=[];i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            if line.startswith('```'):
                i+=1
                while i<len(lines) and not lines[i].strip().startswith('```'):i+=1
                i+=1;continue  # The flow is drawn above; no Mermaid runtime dependency.
            if line.startswith('|'):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    row=lines[i].strip();i+=1
                    if re.fullmatch(r'[|\s:\-]+',row):continue
                    rows.append([inline(x.strip()) for x in row.strip('|').split('|')])
                output.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in rows[0])+'</tr></thead><tbody>')
                for row in rows[1:]:output.append('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>')
                output.append('</tbody></table></div>');continue
            if line.startswith('- ') or re.match(r'^\d+\. ',line):
                ordered=bool(re.match(r'^\d+\. ',line));tag='ol' if ordered else 'ul';output.append('<'+tag+'>')
                while i<len(lines):
                    match=re.match(r'^\d+\. (.*)',lines[i].strip()) if ordered else re.match(r'^- (.*)',lines[i].strip())
                    if not match:break
                    output.append('<li>'+inline(match.group(1))+'</li>');i+=1
                output.append('</'+tag+'>');continue
            if line.startswith('### '):output.append('<h3>'+inline(line[4:])+'</h3>');i+=1;continue
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('|','- ','```','### ')) and not re.match(r'^\d+\. ',lines[i].strip()):paragraph.append(lines[i].strip());i+=1
            output.append('<p>'+inline(' '.join(paragraph))+'</p>')
        return ''.join(output)
    sections=[]
    for part in re.split(r'^## ',text,flags=re.M)[1:]:
        title,_,body=part.partition('\n');sections.append((title,body.splitlines()))
    css='''body{font:16px/1.6 system-ui,sans-serif;margin:0;color:#24333c;background:#f3f5f5}header,main,footer{max-width:1600px;margin:auto;padding:28px}header{padding-bottom:16px}h1{font-size:32px;line-height:1.2;margin:0 0 12px}h2{font-size:24px;line-height:1.3}h3{font-size:19px}p{max-width:1180px}a{color:#175d7d}nav{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}.stage{padding:20px;border-radius:12px;background:white;box-shadow:0 2px 8px #24333c0b}.stage h2{font-size:20px;margin:8px 0}.stage p{margin:8px 0}.number,.tag{display:inline-block;padding:3px 10px;border-radius:20px;background:#e1ece7;color:#284d3f;font-size:13px}.stage .out{font-weight:600}section,details{margin:28px 0}details{background:white;border-radius:12px;padding:20px}summary{cursor:pointer;font-size:20px;font-weight:600}details[open]>summary{margin-bottom:18px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:13px;text-align:left;vertical-align:top;border-bottom:1px solid #dce2e3}th{background:#eaf0f0}.table-wrap{overflow:auto;margin:20px 0}code{font:14px ui-monospace,monospace;background:#edf1f1;padding:2px 4px;border-radius:3px}figure{margin:22px 0;background:white;padding:16px;border-radius:12px}figure img{display:block;width:100%;height:auto}figcaption{font-size:14px;color:#455963;max-width:1150px;margin:12px 0}.note{padding:15px 20px;background:#e8efef;border-radius:10px}.links{display:flex;gap:20px;flex-wrap:wrap}footer{font-size:14px;color:#52636b}li{margin:8px 0}@media(max-width:950px){.flow{grid-template-columns:repeat(2,1fr)}}@media(max-width:550px){.flow{grid-template-columns:1fr}header,main,footer{padding:18px}h1{font-size:27px}}'''
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Terrain generation flow plan</title><style>'+css+'</style></head><body><header><span class="tag">Proposed architecture · four stages · sea level 0 m</span><h1>Terrain generation flow plan</h1><p>What each stage decides, which fields it creates, and how the next stage uses them. Existing selected models are unchanged. The figures below are ideal references and a constructed schematic; neither is a trained generation result.</p><nav><a href="#flow">Flow</a><a href="#fields">First field bundle</a><a href="#examples">Ideal examples</a><a href="#contracts">Stage contracts</a></nav></header><main>'
    cards=[('Land / ocean','Global land fraction + seed','Coast/domain geometry','Calculate coast distance and geometric context.'),
           ('General terrain','Coast + global context','Broad height + relief potential','Allocate broad activity and quiet areas.'),
           ('Structural terrain','All previous fields + context','Height + mountain / lower-region support + local relief','Arrange coherent systems and transitions.'),
           ('Detailed terrain','Structural height + fields','Fine elevation','Calculate slope, normals and other final diagnostics.')]
    page+='<section id="flow"><div class="flow">'
    for i,(title,given,output,extra) in enumerate(cards,1):page+=f'<article class="stage"><span class="number">Stage {i}</span><h2>{title}</h2><p>Input: {given}</p><p class="out">Output: {output}</p><p>{extra}</p></article>'
    page+='</div><p class="note">Fields remain available downstream. Extra channels do not require extra stages. Generate fields that carry additional decisions; calculate geometry already fixed by an output.</p></section>'
    first=next(body for title,body in sections if title.startswith('Recommended first field bundle'))
    page+='<section id="fields"><h2>Recommended first field bundle</h2>'+blocks(first[:next(i for i,l in enumerate(first) if l.startswith('The first implementation'))])+'</section>'
    page+='<section id="examples"><h2>Ideal examples</h2><figure><a href="ideal-earth-stages.png"><img src="ideal-earth-stages.png" alt="Four modern Earth regions at the same physical footprint, shown as coast, broad, structural and detailed reference terrain."></a><figcaption>Same modern Earth terrain at four scales, each panel 1,290.24 km. Candidate spacings: 161.28 km, 40.32 km and 2.88 km. Fixed elevation colours and shading. These illustrate responsibilities; they are not current training targets, sampled outputs or paired PaleoDEM/modern round trips.</figcaption></figure><figure><a href="ideal-field-relationships.png"><img loading="lazy" src="ideal-field-relationships.png" alt="Constructed ideal field relationships with coastline, broad relief potential, mountain and lower-region support, quiet elevated plateau and detailed elevation."></a><figcaption>Constructed design schematic, not Earth or a trained result. Mountain/lower-region supports are authored examples, not a validated terrain-label extractor. Native-resolution fields are labelled. Coast distance here comes from analytic geometry; production requires the actual sampled spherical coastline.</figcaption></figure></section>'
    page+='<section id="contracts"><h2>Detailed stage and field contracts</h2>'
    for title,body in sections:
        page+='<details id="'+slug(title)+'"><summary>'+html.escape(title)+'</summary>'+blocks(body)+'</details>'
    page+='</section><p class="links"><a href="../../docs/terrain-generation-flow-plan.md">Complete Markdown plan</a><a href="provenance.json">Illustration provenance</a><a href="../../docs/terrain-multifield-architecture.md">Earlier four-stage design</a></p></main><footer>Design analysis only. No training, model sampling or planet generation. Zero sea level; no global height shift or forced land coverage.</footer></body></html>'
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    started=time.perf_counter();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(4,4,figsize=(14,13.2),constrained_layout=True)
    provenance={'purpose':'Ideal design references, not model outputs, saved training records, or geographically paired PaleoDEM/modern round trips.',
        'source':'Local ETOPO 2022 30s raster, through existing spherical regional resampling helper.',
        'footprint_km':EXTENT,'fine_spacing_m':2880.,'structural_spacing_m':40320.,'broad_spacing_m':161280.,
        'colours_m':[-6000,6000],'sea_level_m':0.,'shading':'NW 45 degree, 5x slopes at each displayed native spacing.',
        'cases':[],'training':False,'model_inference':False,'selected_changed':False,
        'schematic':'Separate analytically constructed geography; hand-authored mountain/lower-region support. Demonstrates desired relationships, not a validated DEM-label extractor.'}
    source=ROOT/'data/etopo_2022_30s_surface.tif';stat=source.stat();provenance['source_fingerprint']={'path':str(source),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    for row,(key,title,lat,lon) in enumerate(CASES):
        fine=sample('Modern Earth',lat,lon,n=448,spacing=2880.).astype(np.float32)
        if not np.isfinite(fine).all():raise ValueError('Nonfinite reference')
        broad=resample(fine,8);structural=resample(fine,32)
        np.savez_compressed(OUT/f'{key}-ideal-reference.npz',fine_metres=fine,structural_metres=structural,broad_metres=broad,land_mask=broad>0)
        axis(axs[row,0],broad>0,title+'\n1 · Coarse land/ocean reference',categorical=True)
        axis(axs[row,1],broad,title+'\n2 · Broad height / depth',161280.)
        axis(axs[row,2],structural,title+'\n3 · Organized structural height',40320.)
        axis(axs[row,3],fine,title+'\n4 · Detailed elevation reference',2880.)
        provenance['cases'].append({'key':key,'title':title,'latitude':lat,'longitude':lon,'orientation':'north-up spherical aeqd local frame'})
        print(title+' reference rendered',flush=True)
    fig.suptitle('Ideal stage responsibilities illustrated by coarsening the SAME modern Earth terrain\nEach panel: 1,290.24 km. These are supplied references, not generated round trips or current training targets.',fontsize=12)
    legend(fig,axs[:,1:]);fig.savefig(OUT/'ideal-earth-stages.png',dpi=125);plt.close(fig)
    schematic(OUT)
    provenance['elapsed_seconds']=time.perf_counter()-started
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    report(OUT)
    print(f'Design figures complete in {provenance["elapsed_seconds"]:.1f}s; no model inference or training.',flush=True)


if __name__=='__main__':main()
