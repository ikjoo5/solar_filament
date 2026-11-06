"""External data remain evaluation-only, physical-disjoint and temporally grouped."""
from datetime import date,timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu

TOOLS=Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
import external_evaluation_data as external


def write_image(path,value):
    image=np.full((2048,2048),value,np.uint8)
    assert cv2.imwrite(str(path),image)
    return image


def mask(left=10):
    image=np.zeros((2048,2048),np.uint8);image[10:30,left:left+20]=1
    return mu.encode(np.asfortranarray(image))


class ExternalEvaluationTests(unittest.TestCase):
    def test_official_ids_decoded_pixel_duplicates_and_external_rater_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);train=root/'official_train';test=root/'official_test';images=root/'external'
            for path in (train,test,images):path.mkdir()
            write_image(train/'20110101010101Ch.png',10);write_image(test/'20120101010101Bh.png',20)
            write_image(images/'20110101010101Ch.png',30)  # Same physical identifier despite changed pixels.
            write_image(images/'renamed_copy.png',10)  # Different file name; identical decoded pixels.
            write_image(images/'20250101010101Lh.png',40)
            rows=[dict(id='a',file_name='20110101010101Ch.png'),dict(id='b',file_name='renamed_copy.png'),
                dict(id='c1',file_name='20250101010101Lh.png'),dict(id='c2',file_name='20250101010101Lh.png')]
            ann=root/'annotations.json';ann.write_text(json.dumps(dict(images=rows,annotations=[])))
            plan=external.prepare_external_plan(ann,images,train,test)
            self.assertEqual(plan['data_role'],'external_evaluation_only');self.assertFalse(plan['allow_training'])
            self.assertFalse(plan['allow_pseudo_label_training'])
            self.assertEqual(plan['summary']['eligible_physical_images'],1)
            self.assertEqual(plan['summary']['eligible_annotation_groups'],2)
            self.assertEqual(plan['records'][0]['source_image_ids'],['c1','c2'])
            self.assertEqual(plan['summary']['exclusion_reasons'],{'official_physical_identifier':1,'official_file_or_decoded_pixel_duplicate':1})

    def test_temporal_assignment_has_no_label_input_and_embargo_separates_dates(self):
        records=[]
        for i in range(365):
            day=date(2025,1,1)+timedelta(days=i)
            for site in ('ch','bh'):
                records.append(dict(capture_day=day.isoformat(),pixel_sha256=f'{i}:{site}'))
        result=external.assign_external_splits(records)
        by_day={}
        for row in result:
            prior=by_day.setdefault(row['capture_day'],row['split']);self.assertEqual(prior,row['split'])
        dev=[date.fromisoformat(d) for d,s in by_day.items() if s=='external_dev']
        confirm=[date.fromisoformat(d) for d,s in by_day.items() if s=='external_confirm']
        self.assertTrue(dev and confirm)
        self.assertGreater(min(abs((a-b).days) for a in dev for b in confirm),3)
        self.assertTrue(any(s=='external_embargo' for s in by_day.values()))

    def score_plan(self,root):
        source=root/'annotations.json'
        polygon=[10,10,30,10,30,30,10,30]
        source.write_text(json.dumps(dict(images=[dict(id=x,file_name=x+'.png') for x in ('a','b','blank')],
            annotations=[dict(image_id=x,segmentation=[polygon]) for x in ('a','b')])))
        records=[dict(evaluation_key='image1',source_image_ids=['a','b'],split='external_dev',observatory='ch',time_block_27days='1',time_clean_3days=True),
                 dict(evaluation_key='image2',source_image_ids=['blank'],split='external_dev',observatory='bh',time_block_27days='2',time_clean_3days=False)]
        return dict(data_role='external_evaluation_only',allow_training=False,records=records,
            annotations=str(source),annotations_sha256=external.sha(source.read_bytes()))

    def test_official_all_pairs_raters_blank_groups_and_time_clean_strata(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=self.score_plan(Path(directory))
            result=external.score_external(plan,dict(image1=[mask(),mask(100)],image2=[mask(200)]))
            official=result['official_nonblank_pq'];all_groups=result['all_metadata_groups_including_blank_GT']
            self.assertEqual((official['tp_pairs'],official['fp'],official['fn']),(2,2,0))
            self.assertAlmostEqual(official['pq'],2/3)
            self.assertEqual(all_groups['fp'],3);self.assertAlmostEqual(all_groups['pq'],2/3.5)
            self.assertEqual(result['physical_images'],2);self.assertEqual(result['annotation_groups'],3)
            self.assertEqual(result['time_clean_3day_nonblank']['physical_images'],1)
            self.assertFalse(result['training_performed'])

    def test_role_missing_images_changed_labels_and_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=self.score_plan(root);pred=dict(image1=[mask()],image2=[])
            wrong=dict(plan,allow_training=True)
            with self.assertRaises(ValueError):external.score_external(wrong,pred)
            with self.assertRaises(ValueError):external.score_external(plan,{'image1':[mask()]})
            with self.assertRaises(ValueError):external.score_external(plan,dict(image1=[mask(),mask()],image2=[]))
            Path(plan['annotations']).write_text('{}')
            with self.assertRaises(ValueError):external.score_external(plan,pred)

    def test_holdout_not_scored_by_default_and_bad_annotation_reference_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=self.score_plan(root)
            plan['records'][0]['split']='external_confirm'
            result=external.score_external(plan,{'image2':[]})
            self.assertEqual(result['physical_images'],1)
            self.assertEqual(result['official_nonblank_pq']['groups'],0)
            self.assertEqual(result['split'],'external_dev')
            explicit=external.score_external(plan,{'image1':[mask()]},split='external_confirm')
            self.assertEqual(explicit['official_nonblank_pq']['tp_pairs'],2)

    def test_orphan_target_references_cannot_silently_become_blank_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);images=root/'images';images.mkdir()
            source=root/'annotations.json';source.write_text(json.dumps(dict(images=[dict(id='known',file_name='frame.png')],
                annotations=[dict(image_id='unknown',segmentation=[[0,0,1,0,1,1]])])))
            with self.assertRaisesRegex(ValueError,'unknown image ID'):
                external.prepare_external_plan(source,images,root/'train',root/'test')


if __name__=='__main__':unittest.main()
