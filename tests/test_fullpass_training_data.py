"""Coverage, observer consistency, mask clipping and reusable YOLO export contracts."""
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
from pycocotools import mask as mu
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import fullpass_training_data as data


def encoded(mask):
    rle=mu.encode(np.asfortranarray(mask)); area=int(mu.area(rle)); box=mu.toBbox(rle).tolist()
    rle['counts']=rle['counts'].decode()
    return dict(segmentation=rle,area=area,bbox=box)


class Store:
    def __init__(self, names):
        self.hashes={name:'fixture-'+name for name in names}; self.loads=0
    def load(self, name):
        assert name in self.hashes
        self.loads+=1
        return np.full((2048,2048),127,np.uint8)


def fixture_dataset(tmp_path, **kwargs):
    real=dict(images=[dict(id=1,file_name='gong.jpeg',width=2048,height=2048),
        dict(id=2,file_name='gong.jpeg',width=2048,height=2048)],annotations=[
        dict(id=1,image_id=1,segmentation=[[1000,1000,1050,1000,1050,1050,1000,1050]],
            area=2500,bbox=[1000,1000,50,50],spine=[1005,1020,1045,1020]),
        dict(id=2,image_id=2,segmentation=[[50,50,90,50,90,80,50,80]],
            area=1200,bbox=[50,50,40,30],spine=[55,60,85,60])])
    mask=np.zeros((2048,2048),np.uint8)
    mask[2:20,2:20]=1; mask[8:14,8:14]=0; mask[100:105,100:105]=1; mask[2047,2047]=1
    synthetic=dict(images=[dict(id='shape',file_name='shape.png',width=2048,height=2048)],
        annotations=[dict(id='shape_mask',image_id='shape',**encoded(mask))])
    external=np.zeros_like(mask);external[1010:1030,1010:1030]=1
    sunspot=np.zeros_like(mask);sunspot[200:250,200:250]=1
    records=[dict(file_name='swe.jpg',width=2048,height=2048,spine_valid=False,
        annotations=[dict(id='f',category_id=0,**encoded(external)),dict(id='s',category_id=3,**encoded(sunspot))]),
        dict(file_name='spot_only.jpg',width=2048,height=2048,spine_valid=False,
        annotations=[dict(id='s2',category_id=3,**encoded(sunspot))])]
    return data.FullPassTileDataset(real,[1,2],Store(['gong.jpeg']),synthetic,Store(['shape.png']),
        data.mixed.swefil_coco(records),Store(['swe.jpg','spot_only.jpg']),
        seed=42,total_epochs=100,cache_dir=tmp_path/'targets',augment=False,**kwargs)


def test_every_physical_image_and_pixel_appears_once_at_all_epochs(tmp_path):
    dataset=fixture_dataset(tmp_path)
    assert len(dataset)==16
    expected={(domain,name,top,left) for domain,name in dataset.physical_images
        for top,left in data.POSITIONS}
    order=[]
    for epoch in (1,3,4,60,95,100):
        dataset.epoch=epoch
        rows=[dataset.sample_record(i) for i in range(len(dataset))]
        keys=[(r['domain'],r['file_name'],r['top'],r['left']) for r in rows]
        assert len(set(keys))==len(keys)==len(expected)
        assert set(keys)==expected
        assert Counter(r['domain'] for r in rows)=={'real':4,'swefil':8,'synthetic':4}
        for domain,name in dataset.physical_images:
            selected=[r for r in rows if (r['domain'],r['file_name'])==(domain,name)]
            assert len({r['group_id'] for r in selected})==1
            cover=np.zeros((2048,2048),np.uint8)
            for row in selected:cover[row['top']:row['top']+1024,row['left']:row['left']+1024]+=1
            assert np.all(cover==1)
        order.append(keys)
    assert order[0]!=order[1]
    assert dataset.domain_probabilities(1)==dataset.domain_probabilities(100)


def test_observers_rotate_without_merging_and_are_deterministic(tmp_path):
    first=fixture_dataset(tmp_path/'a');second=fixture_dataset(tmp_path/'b')
    g1=first.group_for_image('real','gong.jpeg',1);g2=first.group_for_image('real','gong.jpeg',2)
    assert {g1,g2}=={'1','2'}
    assert first.group_for_image('real','gong.jpeg',3)==g1
    for epoch in (1,2,57):
        first.epoch=second.epoch=epoch
        assert first.epoch_manifest()==second.epoch_manifest()
        for index in (0,5,15):
            a,b=first.sample_record(index),second.sample_record(index)
            assert {k:v for k,v in a.items() if k!='rng'}=={k:v for k,v in b.items() if k!='rng'}
            assert np.array_equal(a['rng'].random(10),b['rng'].random(10))


def test_cropped_masks_reconstruct_image_and_sunspot_only_is_negative(tmp_path):
    dataset=fixture_dataset(tmp_path)
    dataset.epoch=1
    restored={key:np.zeros((2048,2048),np.uint8) for key in dataset.physical_images}
    for index in range(len(dataset)):
        row=dataset.sample_record(index);x,y,valid=dataset[index]
        assert tuple(x.shape)==(3,1024,1024)
        assert tuple(y.shape)==(2,1024,1024)
        assert bool(valid)==(row['domain']=='real')
        top,left=row['top'],row['left']
        restored[row['domain'],row['file_name']][top:top+1024,left:left+1024]=y[0].numpy().astype(np.uint8)
    for (domain,name),mask in restored.items():
        expected,_,_=dataset._target(domain,dataset.group_for_image(domain,name))
        assert np.array_equal(mask,expected)
    assert not restored['swefil','spot_only.jpg'].any()
    assert restored['synthetic','shape.png'][2047,2047]==1


def test_no_crop_budget_domain_quota_or_blank_exclusion(tmp_path):
    for options in ({'crop':512},{'length':400},{'synthetic_ratio':.1},
                    {'swefil_ratio':.6},{'warmup_epochs':3},{'real_finish_epochs':6},
                    {'include_empty_synthetic':False}):
        with pytest.raises(ValueError):fixture_dataset(tmp_path,**options)
    dataset=fixture_dataset(tmp_path)
    for index in (-1,16,1.5,True):
        with pytest.raises(IndexError):dataset.sample_record(index)
    for epoch in (0,-1,1.5,True):
        with pytest.raises(ValueError):dataset.tile_rows(epoch)


def test_yolo_export_preserves_all_variants_tiles_blanks_and_tiny_instances(tmp_path):
    dataset=fixture_dataset(tmp_path/'dataset')
    yaml,manifest=data.export_yolo_dataset(dataset,tmp_path/'yolo')
    assert yaml.is_file()
    assert len((yaml.parent/'unused_validation.txt').read_text().splitlines())==1
    assert manifest['coverage']['tiles_per_epoch']==16
    assert len(manifest['physical_tiles'])==16
    assert len(manifest['records'])==20 # extra observer contributes four label variants only
    assert manifest['polygon_audit']['multipart_instances']>=1
    assert manifest['polygon_audit']['added_pixels']>0 # documented holes/bridge approximation
    assert manifest['polygon_audit']['lost_pixels']==0
    assert manifest['polygon_audit']['blank_variant_tiles']>0
    for record in manifest['records']:
        image,label=Path(record['image']),Path(record['label'])
        assert image.is_symlink() and image.resolve()==Path(record['physical_image'])
        assert cv2.imread(str(image),cv2.IMREAD_GRAYSCALE).shape==(1024,1024)
        if record['file_name']=='spot_only.jpg':assert label.read_text()==''
        if record['file_name']=='shape.png' and record['tile_id']==3:
            assert record['instances']==1
            from chase_yolo_training import polygon_mask
            mask=polygon_mask(label.read_text().strip(),(1024,1024))
            assert mask[1023,1023]
    for epoch in (1,2):
        selected=[r for r in manifest['records'] if r['group_id']==data.group_for_image(
            manifest['variants'][r['domain']][r['file_name']],manifest['seed'],r['domain'],r['file_name'],epoch)]
        assert len(selected)==16
        assert len({(r['physical_key'],r['tile_id']) for r in selected})==16


def test_verified_yolo_cache_reuses_pngs_and_detects_changed_sources(tmp_path):
    dataset=fixture_dataset(tmp_path/'dataset')
    root=tmp_path/'yolo'
    _,manifest=data.export_yolo_dataset(dataset,root)
    for store in dataset.stores.values():store.loads=0
    with patch.object(data,'_write_png',side_effect=AssertionError('Unnecessary tile rebuild')):
        _,again=data.export_yolo_dataset(dataset,root)
    assert manifest==again
    assert all(store.loads==0 for store in dataset.stores.values())
    dataset.stores['real'].hashes['gong.jpeg']='changed-source'
    with pytest.raises(ValueError,match='different sources'):data.export_yolo_dataset(dataset,root)


def test_corrupt_export_label_is_rebuilt_instead_of_trusted(tmp_path):
    dataset=fixture_dataset(tmp_path/'dataset');root=tmp_path/'yolo'
    _,manifest=data.export_yolo_dataset(dataset,root)
    record=next(r for r in manifest['records'] if r['instances'])
    path=Path(record['label']);expected=path.read_text();path.write_text('corrupt')
    _,rebuilt=data.export_yolo_dataset(dataset,root)
    assert path.read_text()==expected
    assert rebuilt['identity']==manifest['identity']


def test_incomplete_manifest_cannot_reduce_coverage_on_reuse(tmp_path):
    dataset=fixture_dataset(tmp_path/'dataset');root=tmp_path/'yolo'
    _,manifest=data.export_yolo_dataset(dataset,root)
    manifest['records'].pop()
    (root/'manifest.json').write_text(json.dumps(manifest))
    _,rebuilt=data.export_yolo_dataset(dataset,root)
    assert len(rebuilt['records'])==20
