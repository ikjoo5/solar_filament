from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import fivefold_full_pipeline as full
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_fivefold_study_evaluation import fixture,save,signed_record


class EmptyProvider:
    def records(self,image,views):
        return [dict(image=image,view=view,shape=[2048,2048],coordinate_space='original',instances=[],max_det_reached=False)
                for view in views]
    def geometric(self,image,pixels):return self.records(image,['base','scale','hflip','vflip'])
    def preprocessing(self,image,pixels):return self.records(image,['limb','clahe16','clahe32','percentile','unsharp','gaussian','bilateral','blackhat'])
    def tiled(self,image,pixels):
        return [dict(self.records(image,[f'tile{s}_y{y}_x{x}'])[0],crop_bbox=[x,y,1024,1024],input_size=s)
                for y in (0,512,1024) for x in (0,512,1024) for s in (1536,2048)]


class FullTests(unittest.TestCase):
    def test_composition_calls_both_frozen_model_branches(self):
        mask=dict(size=[2048,2048],counts=b'test')
        rich=dict(components={'0.5':[dict(area=400,rle=mask)]})
        with patch.object(full,'build_yolo_reference',return_value=[mask]) as yolo, \
             patch.object(full,'build_native_submission',return_value=dict(final=[mask],rich_refined=[mask])) as native:
            result=full.compose('image',[],[],[],rich,variants=False)
        yolo.assert_called_once();native.assert_called_once()
        self.assertEqual(set(result),{'full_selected'})
        self.assertEqual(result['full_selected'][0]['counts'],'test')

    def test_yolo_and_native_must_share_fold_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'fold0/yolo';directory.mkdir(parents=True)
            sources={k:k for k in ('annotations','fold_plan','train_split','val_split','base_runtime')}
            cfg=dict(settings=dict(fold=0,architecture='yolo26s-seg.pt',sources=sources))
            cfg['identity']=full.fingerprint(cfg['settings'])
            meta=dict(identity=cfg['identity'],stage_epoch=30,fold=0,checkpoint='unused',checkpoint_sha256='a'*64)
            (directory/'config.json').write_text(json.dumps(cfg));(directory/'stage030.json').write_text(json.dumps(meta))
            native=dict(training_config=dict(source_sha256=sources))
            self.assertEqual(full.audit_yolo(root,native,30,0,False),meta)
            bad=dict(source_sha256=dict(sources,val_split='other'))
            with self.assertRaisesRegex(ValueError,'sources differ'):full.audit_yolo(root,dict(training_config=bad),30,0,False)

    def test_complete_pipeline_export_cache_and_oof_score_are_connected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ann,plan=fixture(root);images=root/'images';images.mkdir()
            summaries=[]
            for fold in range(5):
                stage=root/f'fold{fold}/stage030';summary=json.loads((stage/'stage_summary.json').read_text())
                path=stage/'rich_records.jsonl';records=[json.loads(line) for line in path.read_text().splitlines()]
                for r in records:
                    cv2.imwrite(str(images/r['file_name']),np.full((2048,2048),100,np.uint8))
                    r['image_sha256']=full.digest(images/r['file_name']);r['size']=[2048,2048];signed_record(r)
                path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                exported=json.loads((stage/'export_summary.json').read_text())
                exported.update(rich_predictions_sha256=full.digest(path),records_sha256={r['file_name']:r['record_sha256'] for r in records})
                save(stage/'export_summary.json',exported)
                summary.update(rich_predictions_sha256=full.digest(path),export=dict(rich_predictions=str(path)))
                summary['export_summary_sha256']=full.digest(stage/'export_summary.json')
                save(stage/'stage_summary.json',summary);summaries.append(summary)
                weight=stage/'yolo.pt';weight.write_bytes(b'owned fixture only')
                yolo=root/f'fold{fold}/yolo';yolo.mkdir()
                settings=dict(fold=fold,architecture='yolo26s-seg.pt',sources={k:summary['training_config']['source_sha256'][k]
                    for k in ('annotations','fold_plan','train_split','val_split','base_runtime')})
                signature=full.fingerprint(settings)
                save(yolo/'config.json',dict(settings=settings,identity=signature))
                save(yolo/'stage030.json',dict(identity=signature,fold=fold,stage_epoch=30,checkpoint=str(weight),checkpoint_sha256=full.digest(weight)))
            save(root/'stage030_progress.json',dict(summaries=summaries))
            for fold in range(5):
                cfg=dict(study_root=str(root),stage_epoch=30,fold=fold,images=str(images))
                result=full.run_export(cfg,_provider=EmptyProvider())
                self.assertEqual(result['methods'],sorted(full.METHODS))
                with patch.object(EmptyProvider,'geometric',side_effect=AssertionError('Cached inference repeated')):
                    full.run_export(dict(cfg,cached_only=True,device='cpu'),_provider=EmptyProvider())
            result=full.evaluate(ann,plan,root,30,root/'result.json')
            self.assertTrue(result['complete_fivefold']);self.assertEqual(result['physical_images'],10)
            self.assertEqual(set(result['results']),set(full.METHODS))
            self.assertIsNotNone(result['results']['full_selected']['pooled_oof'])
            bad=root/'fold0/yolo/stage030.json';payload=json.loads(bad.read_text());payload['fold']=1;save(bad,payload)
            with self.assertRaisesRegex(ValueError,'Wrong YOLO'):full.evaluate(ann,plan,root,30,root/'bad.json')


if __name__=='__main__':unittest.main()
