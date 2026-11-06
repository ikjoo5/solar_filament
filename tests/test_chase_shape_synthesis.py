"""Actual pixel/label, leakage, source-integrity and runtime-reset checks."""
import csv
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import chase_shape_synthesis as synthesis


def solar(size=128, brightness=150):
    yy,xx=np.indices((size,size))
    disk=(xx-size/2)**2+(yy-size/2)**2 <= (size*.44)**2
    texture=((xx*7+yy*3)%5)-2
    values=np.rint(brightness+.06*(xx-size/2)+texture).clip(0,255).astype(np.uint8)
    return values*disk.astype(np.uint8)


def shape(size=128, x=78, y=62):
    mask=np.zeros((size,size),np.uint8)
    cv2.line(mask,(x,y-8),(x,y+8),1,3)
    cv2.line(mask,(x-5,y-3),(x+5,y+3),1,3)
    return mask


def test_measured_attenuation_uses_local_background_and_preserves_joint_conditioning():
    image=solar()
    mask=shape()
    original=image.copy()
    image[mask>0]=np.rint(image[mask>0]*.76).astype(np.uint8)
    measured=synthesis.measure_appearance(image,mask,mask,dict(cx=64,cy=64,radius=56))
    assert .23<measured['center_tau']<.33
    assert abs(measured['background']-np.median(original[mask>0]))<4
    other=dict(measured,background=25,radius=.9,center_tau=.05,area=10000)
    chosen,index,distance=synthesis.choose_appearance([other,measured],150,.25,mask.sum(),np.random.default_rng(2))
    assert index==1 and chosen['center_tau']==measured['center_tau']
    assert distance<.1


def test_renderer_preserves_every_unlabelled_pixel_and_rejects_invalid_placements():
    background=solar()
    local=shape()[50:77,70:87]
    profile=dict(center_tau=.25,edge_ratio=.3,variation=.15,width=5)
    forbidden=np.zeros_like(background,bool);forbidden[30:40,30:40]=True
    args=(background,local,forbidden,70,50,dict(cx=64,cy=64,radius=56),profile)
    output,mask,qa=synthesis.render_at(*args,np.random.default_rng(8))
    np.testing.assert_array_equal(output[mask==0],background[mask==0])
    np.testing.assert_array_equal(output[forbidden],background[forbidden])
    assert qa['visible_fraction']==1 and np.all(output[mask>0]<background[mask>0])
    assert np.array_equal(synthesis.render_at(*args,np.random.default_rng(8))[0],output)
    forbidden[mask>0]=True
    with pytest.raises(ValueError,match='protected foreground'):
        synthesis.render_at(*args,np.random.default_rng(8))
    with pytest.raises(ValueError,match='outside image'):
        synthesis.render_at(background,local,forbidden,-1,50,args[5],profile,np.random.default_rng(8))
    with pytest.raises(ValueError,match='invisible'):
        synthesis.render_at(background,local,np.zeros_like(forbidden),70,50,args[5],dict(profile,center_tau=0),np.random.default_rng(8))


def test_canonical_rle_preserves_branches_holes_and_disconnected_geometry():
    mask=shape()
    mask[64,78]=0
    mask[30:32,30:32]=1
    encoded=synthesis.encode(mask)
    assert isinstance(encoded['counts'],str)
    np.testing.assert_array_equal(synthesis.decode(encoded),mask)
    ann=synthesis._annotations([mask],'example')[0]
    assert ann['area']==mask.sum() and ann['bbox']==[30.,30.,56.,43.]


def test_real_gong_rgb_storage_uses_archived_grayscale_decode(tmp_path):
    path=tmp_path/'real.jpg'
    rgb=cv2.cvtColor(solar(),cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path),rgb)
    actual=synthesis.read_gray(path,128,synthesis.digest(path))
    np.testing.assert_array_equal(actual,cv2.imread(str(path),cv2.IMREAD_GRAYSCALE))
    high_depth=tmp_path/'uint16.png';cv2.imwrite(str(high_depth),solar().astype(np.uint16)*100)
    with pytest.raises(ValueError,match='uint8'):synthesis.read_gray(high_depth,128)


@pytest.fixture
def experiment(tmp_path):
    images=tmp_path/'real';images.mkdir()
    clean=tmp_path/'clean_solar.png';synthesis.atomic_png(clean,solar(brightness=156))
    doc=dict(images=[],annotations=[])
    rows=[]
    for i in range(2):
        name=f'train{i}.png';image=solar(brightness=145+i*5)
        mask=shape(x=78+i*3,y=62+i*2)
        image[mask>0]=np.rint(image[mask>0]*.77).astype(np.uint8)
        synthesis.atomic_png(images/name,image)
        for rater in range(2 if i==0 else 1):
            group=f't{i}_{rater}'
            doc['images'].append(dict(id=group,file_name=name,width=128,height=128))
            doc['annotations'].append(dict(id=group+'_ann',image_id=group,segmentation=synthesis.encode(mask)))
            rows.append(dict(image_id=group,file_name=name))
    doc['images'].append(dict(id='held_out',file_name='absent_validation.png',width=128,height=128))
    doc['annotations'].append(dict(id='never_use',image_id='held_out',segmentation=[]))
    annotations=tmp_path/'annotations.json';synthesis.atomic_json(annotations,doc)
    def split(name,values):
        path=tmp_path/name
        with path.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['image_id','file_name']);writer.writeheader();writer.writerows(values)
        return path
    train=split('train.csv',rows)
    val=split('val.csv',[dict(image_id='held_out',file_name='absent_validation.png')])
    donors=[]
    for number in range(2):
        image_path=tmp_path/f'donor{number}.png';mask_path=tmp_path/f'mask{number}.png'
        mask=shape(x=66+number*3,y=62)
        # Keep one tiny component, a separate branch and a manually ignored one.
        mask[20,20]=1
        mask[95:102,80:85]=1
        synthesis.atomic_png(image_path,solar(brightness=133+number))
        synthesis.atomic_png(mask_path,mask*255)
        donors.append(dict(id=f'donor{number}',date='2024-01-01',image_path=str(image_path),
                           mask_path=str(mask_path),image_sha256=synthesis.digest(image_path),
                           mask_sha256=synthesis.digest(mask_path)))
    report=tmp_path/'source_report.json';synthesis.atomic_json(report,dict(records=donors))
    chase_manifest=tmp_path/'chase_manifest.json';synthesis.atomic_json(chase_manifest,dict(fixture=True))
    def provider(*_):return donors,report
    def aligned(record):
        image=cv2.imread(record['image_path'],0)
        mask=(cv2.imread(record['mask_path'],0)>0).astype(np.uint8)
        valid=np.ones_like(mask);valid[95:102,80:85]=0
        return image,mask,valid
    cfg=dict(images=str(images),annotations=str(annotations),train_split=str(train),val_split=str(val),
        chase_root=str(tmp_path/'chase'),chase_manifest=str(chase_manifest),cache_root=str(tmp_path/'cache'),
        output_root=str(tmp_path/'output'),seed=42,clean_background=str(clean),
        clean_background_provenance='external_user_supplied',
        clean_background_source_note='Fixture user-declared clean external image')
    hashes={key:synthesis.digest(cfg[key]) for key in ('annotations','train_split','val_split')}
    def run():
        return synthesis._prepare_synthesis(cfg,expected_hashes=hashes,expected_counts=(2,1),size=128,
                chase_provider=provider,aligned_provider=aligned,expected_chase=2)
    return cfg,run,donors


def test_clean_synthesis_labels_are_visible_disjoint_and_original_sources_are_readonly(experiment):
    cfg,run,donors=experiment
    paths=list(Path(cfg['images']).glob('*.png'))+[Path(cfg['clean_background'])]
    before={str(p):synthesis.digest(p) for p in paths}
    result=run()
    assert {str(p):synthesis.digest(p) for p in paths}==before
    doc=json.loads(Path(result['annotations_path']).read_text())
    assert len(doc['images'])==2 and len(doc['annotations'])==6
    assert 'held_out' not in json.dumps(doc)
    clean,_,_=synthesis.read_clean(cfg['clean_background'],128)
    for record in doc['images']:
        pixels=cv2.imread(str(Path(result['images_dir'])/record['file_name']),0)
        masks=[synthesis.decode(a['segmentation']) for a in doc['annotations'] if a['image_id']==record['id']]
        summed=sum(masks)
        assert summed.max()==1
        np.testing.assert_array_equal(pixels[summed==0],clean[summed==0])
        assert np.mean(pixels[summed>0]<clean[summed>0])>=.9
    manifest=synthesis.load_envelope(result['manifest_path'],result['identity'])
    assert manifest['unique_physical_backgrounds']==1
    assert manifest['synthetic_instances']==6
    assert not manifest['validation_pixels_used_for_calibration_or_training']
    assert len(result['preview_paths'])==3  # Includes true source full disk.
    inventory=synthesis.load_envelope(Path(result['manifest_path']).with_name('shape_inventory.json'),result['identity'])
    assert all(item['explicit_or_unobserved_exclusions']==1 for item in inventory)
    assert all(len(item['shapes'])==2 for item in inventory)  # Tiny component was not called overlap.


def test_local_cache_reset_and_corruption_reconstruct_exact_saved_data(experiment):
    cfg,run,_=experiment
    initial=run();identity=initial['identity']
    annotation_bytes=Path(initial['annotations_path']).read_bytes()
    before={p.name:synthesis.digest(p) for p in Path(initial['images_dir']).glob('*.png')}
    first=next(Path(initial['images_dir']).glob('*.png'));first.write_bytes(b'bad image cache')
    repaired=run()
    assert synthesis.digest(first)==before[first.name]
    shutil.rmtree(cfg['cache_root'])
    replay=run()
    assert replay['identity']==identity
    assert Path(replay['annotations_path']).read_bytes()==annotation_bytes
    assert {p.name:synthesis.digest(p) for p in Path(replay['images_dir']).glob('*.png')}==before


def test_changed_clean_source_or_persistent_statistics_cannot_silently_resume(experiment):
    cfg,run,_=experiment
    result=run()
    stats=Path(result['statistics_path']);original=stats.read_text()
    edited=json.loads(original);edited['payload']['profiles'][0]['center_tau']+=.1
    stats.write_text(json.dumps(edited))
    with pytest.raises(ValueError,match='metadata changed'):run()
    stats.write_text(original)
    pixels=cv2.imread(cfg['clean_background'],0);pixels[64,64]+=1
    synthesis.atomic_png(cfg['clean_background'],pixels)
    with pytest.raises(ValueError,match='inputs/code changed'):run()


def test_heldout_duplicate_and_missing_provenance_are_rejected(experiment):
    cfg,run,_=experiment
    cfg['clean_background_provenance']='unknown'
    with pytest.raises(ValueError,match='provenance'):run()
    cfg['clean_background_provenance']='external_user_supplied'
    shutil.copyfile(cfg['clean_background'],Path(cfg['images'])/'absent_validation.png')
    with pytest.raises(ValueError,match='held-out validation'):run()


def test_obviously_contaminated_clean_background_is_not_treated_as_empty():
    image=solar();image[60:65,60:70]=4
    profile=dict(center_tau=.2,area=30)
    with pytest.raises(ValueError,match='unlabelled dark feature'):
        synthesis.clean_quality(image,dict(cx=64,cy=64,radius=56),[profile])


def test_public_entrypoint_does_not_accept_fixture_hash_bypass(experiment):
    cfg,_,_=experiment
    cfg['expected_hashes']={key:synthesis.digest(cfg[key]) for key in synthesis.EXPECTED}
    with pytest.raises(ValueError,match='original best84'):
        synthesis.prepare_synthesis(cfg)
