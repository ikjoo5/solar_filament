"""Render the README's scientific figures from observations and saved predictions.

No inference, synthetic masks, recoloring of solar intensities, or training occurs.
Supply local paths in a JSON file; raw data and full prediction exports stay outside
Git. See docs/assets/readme/README.md for the input contract and image credits.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pycocotools import mask as mu

from submission_inference import limb, preprocess_variant

SWE = '20100604185114Th'
OBS = '20120330063134Lh'
FINAL_SHA = '0b651ca12822c740e7bdc06ea5f76e700c071a7ef660711b225760f378b5278b'
BG, INK, MUTED = '#f6f8fb', '#15243a', '#526175'
TEAL, GOLD, BLUE = '#13b8a6', '#f7b343', '#53a6ef'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK})


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gray(path):
    value=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if value is None: raise ValueError('Unreadable image: '+str(path))
    return value


def decode(rle):
    value=dict(rle)
    if isinstance(value['counts'],str): value['counts']=value['counts'].encode()
    result=mu.decode(value).astype(bool)
    if result.shape!=(2048,2048): raise ValueError('Expected native 2048 masks')
    return result


def canvas(title, subtitle, labels, width=13.5):
    fig,axes=plt.subplots(1,len(labels),figsize=(width,4.75),facecolor=BG)
    fig.subplots_adjust(left=.018,right=.982,top=.77,bottom=.085,wspace=.035)
    fig.text(.025,.955,title,fontsize=22,fontweight='bold',va='top')
    fig.text(.025,.865,subtitle,fontsize=11,color=MUTED,va='top')
    for ax,label in zip(axes,labels):
        ax.set_title(label,fontsize=12.5,fontweight='bold',pad=10)
        ax.set_axis_off()
    return fig,axes


def show(ax,image):
    ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')


def overlay(ax,mask,color,alpha=.32):
    from matplotlib.colors import to_rgba
    rgba=np.zeros((*mask.shape,4),np.float32);rgba[mask]=to_rgba(color,alpha)
    ax.imshow(rgba,interpolation='nearest')
    if mask.any() and not mask.all(): ax.contour(mask.astype(float),levels=[.5],colors=[color],linewidths=1.1)


def save(fig,path):
    fig.savefig(path,dpi=120,facecolor=BG,pil_kwargs={'quality':92,'optimize':True})
    plt.close(fig)


def csv_masks(path):
    result=[]
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['filament_id'].rsplit('_',1)[0]==OBS:
                result.append((row['filament_id'],decode(dict(size=[2048,2048],counts=row['segmentation_rle']))))
    return result


def build(inputs,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths={key:Path(value) for key,value in inputs.items()}
    manifest=json.loads(paths['swefil_manifest'].read_text())
    record=next(row for row in manifest['records'] if row['observation_id']==SWE)
    source=paths['swefil_images']/Path(record['file_name']).name
    if digest(source)!=record['image_sha256']: raise ValueError('SWEFil image checksum differs')
    image=gray(source)
    annotation=next(ann for ann in record['annotations'] if ann['id']==1)
    target=decode(annotation['segmentation'])
    # Fixed editorial crop: selected for a visible long filament, not a model score.
    x0,y0,x1,y1=560,110,1240,790
    fig,axes=canvas('What is a solar filament?',
        'Dark, thread-like structures seen against the bright solar disk.',
        ['The whole Sun','A closer look','The region to outline'])
    show(axes[0],image);axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=TEAL,linewidth=2))
    crop=image[y0:y1,x0:x1];show(axes[1],crop);show(axes[2],crop)
    overlay(axes[2],target[y0:y1,x0:x1],TEAL)
    fig.text(.025,.025,'SWEFil / GONG observation · teal = human reference annotation, not a model prediction.',fontsize=10,color=MUTED)
    save(fig,output/'filament_intro.jpg')

    raw=gray(paths['swefil_raw']);corrected=limb(raw);contrast=preprocess_variant(corrected,'clahe16')
    fig,axes=canvas('Prepare the image before looking for filaments',
        'The same observation and display range in every panel.',
        ['Original observation','Balance center-to-edge brightness','One extra YOLO contrast view'])
    for ax,value in zip(axes,(raw,corrected,contrast)): show(ax,value)
    fig.text(.025,.025,'Actual preprocessing from this repository · the extra contrast view is used by YOLO, not by both models.',fontsize=10,color=MUTED)
    save(fig,output/'preprocessing.jpg')

    tiles=json.loads((paths['pixel_tiles']/'manifest.json').read_text())
    image_info=tiles['images'][OBS]
    native_manifest=json.loads((paths['native_predictions'].parent/'manifest.json').read_text())
    expected=native_manifest['completed'][OBS+'.jpeg']
    if image_info['raw_sha256']!=expected['image_sha256']: raise ValueError('Pixels and native predictions have different source images')
    patch=np.zeros((384,384),np.uint8);covered=np.zeros_like(patch,bool)
    used=[]
    for row in image_info['tiles']:
        x,y=row['x'],row['y']
        if x not in (384,512,640) or y not in (384,512,640): continue
        path=paths['pixel_tiles']/row['path']
        if digest(path)!=row['sha256']: raise ValueError('Pixel tile checksum differs')
        tile=gray(path)
        if tile.shape!=(128,128): raise ValueError('Unexpected pixel tile dimensions')
        patch[y-384:y-256,x-384:x-256]=tile;covered[y-384:y-256,x-384:x-256]=True
        used.append(dict(path=row['path'],sha256=row['sha256']))
    if len(used)!=9 or not covered.all(): raise ValueError('Display crop has missing or repeated pixels')
    native=None
    for line in paths['native_predictions'].open('rb'):
        value=json.loads(line)
        if value.get('file_name')==OBS+'.jpeg':
            key=hashlib.sha256(line).hexdigest()
            # The exporter hashes each exact JSONL record, including its newline.
            if key!=expected['record_sha256']: raise ValueError('Native prediction record checksum differs')
            native=value;break
    if native is None or native['image_sha256']!=image_info['raw_sha256']: raise ValueError('Native record missing or mismatched')
    if digest(paths['final_csv'])!=FINAL_SHA: raise ValueError('Expected the reviewed baseline final CSV')
    yolo=csv_masks(paths['yolo_csv'])
    foreground=[('native_'+str(i),decode(a['rle'])) for i,a in enumerate(native['tta']['components']['0.5']) if a['area']>=300]
    final=csv_masks(paths['final_csv'])
    selected=[]
    fig,axes=canvas('Two independent models, one final set of masks',
        'Real saved predictions for the same validation crop. Each final color marks one predicted filament.',
        ['Input crop','YOLO candidates','U-Net candidates','Final combined masks'],width=14.8)
    for ax in axes: show(ax,patch)
    for ax,masks,palette in ((axes[1],yolo,[BLUE]),(axes[2],foreground,[GOLD]),(axes[3],final,[TEAL,'#ed7ba7','#a99bf6'])):
        visible=[]
        for name,mask in masks:
            cropped=mask[384:768,384:768]
            if cropped.any():
                color=palette[len(visible)%len(palette)]
                overlay(ax,cropped,color)
                visible.append(name)
        selected.append(visible)
    fig.text(.025,.025,'GONG / MAGFiLO validation observation · 2012-03-30 · limb-corrected pixels · archived YOLO + best84 baseline.',fontsize=10,color=MUTED)
    save(fig,output/'model_predictions.jpg')
    provenance=dict(
        description='Scientific displays of existing observations, annotations and archived predictions; no new inference.',
        introduction=dict(observation=SWE,image_sha256=digest(source),annotation_id=1,
            crop_xyxy=[x0,y0,x1,y1],swefil_revision=manifest['revision'],raw_image_sha256=digest(paths['swefil_raw'])),
        processing=dict(code='tools/submission_inference.py',code_sha256=digest(Path(__file__).with_name('submission_inference.py')),
            methods=['limb','preprocess_variant:clahe16'],display_range=[0,255]),
        prediction_example=dict(observation=OBS,source_image_sha256=image_info['raw_sha256'],crop_xyxy=[384,384,768,768],
            native_record_sha256=expected['record_sha256'],native_checkpoint_sha256=native_manifest['provenance']['checkpoint_sha256'],
            yolo_csv_sha256=digest(paths['yolo_csv']),final_csv_sha256=digest(paths['final_csv']),
            native_display_threshold=.5,native_display_min_area=300,visible_instances=dict(yolo=selected[0],native=selected[1],final=selected[2]),
            pixel_tiles=used),
        files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(output.glob('*.jpg'))})
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(files=provenance['files'],visible_instances=provenance['prediction_example']['visible_instances']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True,help='Local JSON paths; see asset documentation')
    parser.add_argument('--output',type=Path,default=Path('docs/assets/readme'))
    args=parser.parse_args();build(json.loads(args.inputs.read_text()),args.output)
