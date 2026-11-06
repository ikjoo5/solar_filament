"""Data integrity checks for train-only targeted synthetic generation."""
import csv
from pathlib import Path
import sys

import cv2
import numpy as np
import unittest
import tempfile
import json
import shutil
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import targeted_synthetic_data as data

cv2.setNumThreads(1)


def write_split(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['image_id','file_name']);writer.writerows(rows)


class TargetedSyntheticDataTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.tmp=Path(self.folder.name)

    def test_split_preserves_all_raters_and_rejects_filename_leak(self):
        tmp_path=self.tmp
        doc={'images':[{'id':'a','file_name':'train.jpeg'},{'id':'b','file_name':'train.jpeg'},
                       {'id':'c','file_name':'val.jpeg'}]}
        train,val=tmp_path/'train.csv',tmp_path/'val.csv'
        write_split(train,[('a','train.jpeg'),('b','train.jpeg')]);write_split(val,[('c','val.jpeg')])
        roles=data.split_roles(doc,train,val,expected=(1,1))
        assert roles['train']['ids']=={'a','b'}
        write_split(train,[('a','train.jpeg')]);write_split(val,[('b','train.jpeg'),('c','val.jpeg')])
        with self.assertRaisesRegex(ValueError,'siblings'):data.split_roles(doc,train,val,expected=None)


    def test_polygon_roundtrip_keeps_thin_native_pixels(self):
        mask=np.zeros((2048,2048),np.uint8)
        mask[1001:1004,1223:1378]=1
        line,quality=data.mask_to_yolo(mask)
        # Ultralytics polygon2mask casts float pixel vertices to int32 (no rint).
        points=np.asarray(line.split()[1:],np.float32).reshape(-1,2)*2048
        recovered=np.zeros_like(mask);cv2.fillPoly(recovered,[points.astype(np.int32)],1)
        assert np.array_equal(mask,recovered)
        assert quality['iou']==1 and quality['source_area']==465


    def test_disconnected_parts_remain_one_instance(self):
        mask=np.zeros((128,128),np.uint8);mask[10:35,10:30]=1;mask[40:65,40:60]=1
        line,quality=data.mask_to_yolo(mask)
        assert '\n' not in line and line.startswith('0 ')
        assert quality['connected_contours']==2 and quality['iou']>.95


    def test_hole_error_is_not_silently_called_exact(self):
        mask=np.ones((30,30),np.uint8);mask[5:25,5:25]=0
        with self.assertRaisesRegex(ValueError,'raster IoU'):data.mask_to_yolo(mask)


    def test_dark_residual_generation_is_deterministic_and_aligned(self):
        image=np.full((2048,2048),145,np.uint8);mask=np.zeros_like(image)
        cv2.ellipse(mask,(1024,1024),(25,9),15,0,360,1,-1);image[mask>0]=115
        a=data.make_donor_patch(image,mask,np.random.default_rng(7))
        b=data.make_donor_patch(image,mask,np.random.default_rng(7))
        assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1]) and a[2]==b[2]
        assert np.all(a[0][a[1]==0]==0) and 300<=a[1].sum()<=5000


    def test_donor_without_visible_contrast_is_rejected(self):
        image=np.full((2048,2048),145,np.uint8);mask=np.zeros_like(image);mask[1000:1020,1000:1020]=1
        with self.assertRaisesRegex(ValueError,'dark donor contrast'):
            data.make_donor_patch(image,mask,np.random.default_rng(7))


    def test_paste_preserves_background_foreground_and_exact_support(self):
        image=np.full((2048,2048),145,np.uint8);known=np.zeros_like(image,dtype=bool)
        known[1040:1060,1040:1060]=True;image[known]=101
        support=np.ones((20,20),np.uint8);residual=np.full(support.shape,-.08,np.float32)
        out,mask,quality=data.paste_residual(image,residual,support,known,990,990)
        assert np.array_equal(out[known],image[known])
        assert np.array_equal(out[mask==0],image[mask==0])
        assert mask.sum()==400 and quality['median_darkening_DN']>=1
        with self.assertRaisesRegex(ValueError,'overlaps'):
            data.paste_residual(image,residual,support,known,1045,1045)


    def test_paste_rejects_outside_sun_and_out_of_frame(self):
        image=np.full((2048,2048),145,np.uint8);known=np.zeros_like(image,dtype=bool)
        support=np.ones((20,20),np.uint8);residual=np.full(support.shape,-.08,np.float32)
        with self.assertRaisesRegex(ValueError,'rim'):data.paste_residual(image,residual,support,known,0,0)
        with self.assertRaisesRegex(ValueError,'outside image'):data.paste_residual(image,residual,support,known,2040,1000)


    def test_cache_validates_pixels_labels_and_refuses_changed_sources(self):
        tmp_path=self.tmp
        image=tmp_path/'images'/'one.png';label=tmp_path/'labels'/'one.txt';meta=tmp_path/'meta.json';calls=[]
        def build():
            calls.append(True);pixels=np.full((64,64),100,np.uint8);mask=np.zeros_like(pixels);mask[10:30,20:40]=1
            return pixels,[mask],{'donor':'train_only'}
        expected={'source':'abc'}
        first=data._cached_record(image,label,meta,expected,build)
        second=data._cached_record(image,label,meta,expected,build)
        assert len(calls)==1 and first==second
        label.write_text('corrupt')
        third=data._cached_record(image,label,meta,expected,build)
        assert len(calls)==2 and third==first
        image.write_bytes(b'broken image')
        data._cached_record(image,label,meta,expected,build)
        assert len(calls)==3
        with self.assertRaisesRegex(ValueError,'identity'):
            data._cached_record(image,label,meta,{'source':'changed'},build)


    def test_rle_preserves_authoritative_mask(self):
        mask=np.zeros((2048,2048),np.uint8);mask[500:503,900:1100]=1
        encoded=data.rle(mask)
        assert np.array_equal(data.annotation_mask({'segmentation':encoded}),mask)

    def test_complete_prepare_pair_and_vm_cache_loss_rebuild(self):
        # Full real/image/label/synthetic pipeline on a deliberately tiny fixture.
        # Only expected metadata hashes/counts are replaced; pixel processing is real.
        images=self.tmp/'images';synthetics=self.tmp/'v2';images.mkdir();synthetics.mkdir()
        records=[{'id':'a1','file_name':'a.jpeg'},{'id':'a2','file_name':'a.jpeg'},
                 {'id':'b','file_name':'b.jpeg'},{'id':'v','file_name':'val.jpeg'}]
        polygon=[990,1000,1045,1000,1045,1013,990,1013]
        annotations=[{'id':'donor_'+row['id'],'image_id':row['id'],'segmentation':[polygon]} for row in records]
        doc={'images':records,'annotations':annotations}
        for name in ['a.jpeg','b.jpeg']:
            pixels=np.full((2048,2048),145,np.uint8)
            mask=data.annotation_mask({'segmentation':[polygon]});pixels[mask>0]=110
            cv2.imwrite(str(images/name),pixels,[cv2.IMWRITE_JPEG_QUALITY,100])
        # No validation image is created: any accidental read would fail.
        v2doc={'images':[{'id':1,'file_name':'s0.png'},{'id':2,'file_name':'s1.png'}],
               'annotations':[{'id':'s'+str(i),'image_id':i,'segmentation':[polygon]} for i in [1,2]]}
        for name in ['s0.png','s1.png']:
            cv2.imwrite(str(synthetics/name),np.full((2048,2048),135,np.uint8))
        config={'images':str(images),'synthetic_images':str(synthetics),'cache_root':str(self.tmp/'local'),
                'output_root':str(self.tmp/'drive'),'seed':42,'synthetic_samples':2}
        for key,value in [('annotations',doc),('synthetic_annotations',v2doc),
                          ('synthetic_split',{'train_image_ids':['a1','a2','b'],'val_image_ids':['v']})]:
            path=self.tmp/(key+'.json');path.write_text(json.dumps(value));config[key]=str(path)
        for key,rows in [('train_split',[('a1','a.jpeg'),('a2','a.jpeg'),('b','b.jpeg')]),('val_split',[('v','val.jpeg')])]:
            path=self.tmp/(key+'.csv');write_split(path,rows);config[key]=str(path)
        expected={key:data.digest(config[key]) for key in data.EXPECTED}
        hard=self.tmp/'hard.json';hard.write_text(json.dumps({'annotation_ids':['donor_a1','donor_a2','donor_b'],
            'annotations_sha256':expected['annotations'],'train_split_sha256':expected['train_split']}));config['hard_donors']=str(hard)
        split_roles=data.split_roles
        with patch.object(data,'EXPECTED',expected),patch.object(data,'split_roles',side_effect=lambda d,t,v:split_roles(d,t,v,expected=(2,1))):
            first=data.prepare_dataset(config)
            second=data.prepare_dataset(config)
            assert first['identity']==second['identity']
            assert first['arms']['control_v2']['training_records']==first['arms']['targeted_v3']['training_records']==5
            manifest=json.loads(Path(first['manifest']).read_text())
            assert manifest['training_physical_images']==2 and manifest['validation_used_for_training'] is False
            for item in manifest['records']:
                assert item['annotation_instances']==1 if item['domain']!='v3' else item['annotation_instances']==3
            shutil.rmtree(config['cache_root'])
            third=data.prepare_dataset(config)
            assert third['identity']==first['identity']
            assert Path(third['preview']).is_file()

if __name__=="__main__":
    unittest.main()
