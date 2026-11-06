"""Focused agreement/lineage tests; no neural model or hidden test GT is used."""
from contextlib import redirect_stdout
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import public_reference_audit as audit


def rect(top=10, left=10, height=10, width=10):
    pixels = np.zeros((2048, 2048), np.uint8)
    pixels[top:top+height, left:left+width] = 1
    return mu.encode(np.asfortranarray(pixels))


def plain(mask):
    return dict(size=mask['size'], counts=mask['counts'].decode('ascii'))


def csv_file(path, groups):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['filament_id', 'segmentation_rle'])
        for stem, masks in groups.items():
            for index, mask in enumerate(masks, 1):
                writer.writerow([f'{stem}_{index}', mask['counts'].decode('ascii')])


def raw(mask, source='geometric', view='base'):
    return dict(source=source, view=view, confidence=.9, threshold=None, index=0, mask=mask)


def cache_fixture(root):
    root.mkdir(); mask = rect()
    image = dict(name='a.jpeg', stem='a', sha256='image_sha')
    protocol = dict(images=[image], sources={})
    identity = audit.fingerprint(protocol)
    (root/'protocol.json').write_text(json.dumps(dict(protocol=protocol, identity=identity)))
    runtime_payload = dict(identity=identity, facts={'device':'fixture_cpu'})
    runtime_signature = audit.fingerprint(runtime_payload)
    (root/'runtime_device.json').write_text(json.dumps(dict(payload=runtime_payload, sha256=runtime_signature)))
    empty = dict(size=[2048, 2048], counts='PPPP4')
    branch = dict(components={key: [] for key in ('0.2','0.35','0.5','0.65','0.8')},
                  unions={'foreground':{'0.35':empty}})
    native = dict(branch, tta=branch, file_name='a.jpeg', size=[2048,2048],
                  identity=identity, image_sha256='image_sha')
    views = {stage: [dict(image='a', shape=[2048,2048], coordinate_space='original',
        view=str(i), instances=[]) for i in range(count)] for stage, count in
        (('geometric',4),('preprocessing',8),('tiled',18))}
    def save(stage, data):
        payload = dict(identity=identity,image='a',image_sha256='image_sha',stage=stage,
                       runtime_signature=runtime_signature,data=data)
        envelope = dict(payload=payload,sha256=audit.fingerprint(payload))
        path = root/'cache'/'a'/(stage+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(gzip.compress(audit.canonical(envelope)))
        return envelope['sha256']
    supports = {stage:save(stage, native if stage=='native' else views[stage]) for stage in audit.STAGES}
    baseline_sha = save('baseline',dict(masks=[plain(mask)],support_sha256=supports))
    save('final',dict(masks=[plain(mask)],support_sha256=supports,baseline_sha256=baseline_sha))
    for name in ('submission_pipeline_baseline.csv','submission_pipeline_R1.csv'):
        csv_file(root/name,{'a':[mask]})
    manifest = dict(status='complete',identity=identity,official_images=1,
        output_sha256={name:audit.file_sha(root/name) for name in ('submission_pipeline_baseline.csv','submission_pipeline_R1.csv')})
    (root/'run_manifest.json').write_text(json.dumps(manifest))
    return image, identity, runtime_signature, mask


class ReferenceAgreementTests(unittest.TestCase):
    def test_strict_boundary_and_union_not_instance(self):
        reference = rect(); half = rect(width=5)
        rows, _, _ = audit.analyze_image('a',[('a_1',reference)],[],[],[raw(half)],
                                       {'base':reference,'tta':reference})
        self.assertEqual(rows[0]['raw_max_iou'],.5)
        self.assertEqual(rows[0]['category'],'no_similar_raw_instance_in_saved_pool')
        self.assertEqual(rows[0]['native_base035_reference_pixel_coverage'],1)
        self.assertEqual(rows[0]['matching_yolo_views_deduplicated'],0)

    def test_raw_available_final_missing_and_transitions(self):
        mask=rect()
        rows,_,_=audit.analyze_image('a',[('a_1',mask)],[],[],[raw(mask)],
            replay={'selected':[mask],'filtered':[]})
        self.assertEqual(rows[0]['category'],'raw_similar_final_different')
        self.assertEqual(rows[0]['similarity_loss_transitions'],'selected->filtered')

    def test_partial_overlap_selection_opportunity_is_preserved(self):
        reference=rect(); partial=rect(width=4)
        rows,_,summary=audit.analyze_image('a',[('a_1',reference)],[('a_1',partial)],
            raw=[raw(reference)])
        self.assertEqual(rows[0]['category'],'shape_or_grouping_difference')
        self.assertTrue(rows[0]['raw_similar_but_final_not_similar'])
        self.assertEqual(summary['all_raw_similar_but_final_not_similar'],1)

    def test_aggregate_raw_pool_is_not_an_earlier_stage(self):
        mask=rect()
        rows,_,_=audit.analyze_image('a',[('a_1',mask)],[],[],[raw(mask,source='native')],
            replay={'yolo_consensus':[],'baseline_final':[]})
        self.assertEqual(rows[0]['category'],'raw_similar_final_different')
        self.assertEqual(rows[0]['similarity_loss_transitions'],'')

    def test_audit_and_replay_dependencies_do_not_import_torch(self):
        tools=Path(audit.__file__).parent
        code='import public_reference_audit, public_reference_stage_trace, submission_yolo_recipe, submission_native_recipe, sys; assert "torch" not in sys.modules'
        subprocess.run([sys.executable,'-c',code],cwd=tools,check=True,capture_output=True,text=True)

    def test_exact_duplicates_do_not_inflate_view_votes(self):
        first,second=rect(),rect(width=11)
        rows=[raw(first,view='base'),raw(first,view='flip'),raw(second,view='base'),raw(first,source='native')]
        self.assertEqual(audit.distinct_view_support(rows),2)
        self.assertEqual(audit.distinct_view_support([raw(first,view='base'),raw(first,view='flip')]),1)

    def test_fallback_exact_pixels_and_unknown_own_masks(self):
        square=rect(1000,1000,48,48);stem='20160513133714Th'
        self.assertTrue(audit.known_fallback(stem,square))
        self.assertFalse(audit.known_fallback('other',square))
        rows,ours,_=audit.analyze_image(stem,[(stem+'_1',square)],[(stem+'_1',rect())])
        self.assertEqual(rows[0]['category'],'known_placeholder_excluded')
        self.assertIsNone(ours[0]['reference_max_iou'])

    def test_empty_lists_and_unavailable_are_distinct(self):
        mask=rect()
        unavailable,_,_=audit.analyze_image('a',[('a_1',mask)],[])
        available,_,_=audit.analyze_image('a',[('a_1',mask)],[],[],[])
        self.assertIsNone(unavailable[0]['raw_max_iou'])
        self.assertEqual(available[0]['raw_max_iou'],0)
        rows,ours,summary=audit.analyze_image('a',[],[])
        self.assertEqual((rows,ours),([],[]));self.assertEqual(summary['reference_instances'],0)

    def test_csv_mode_preserves_inputs_and_accounts_for_blank_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ref=root/'ref.csv';base=root/'base.csv';mask=rect()
            csv_file(ref,{'a':[mask]});csv_file(base,{})
            before={p:p.read_bytes() for p in (ref,base)}
            with redirect_stdout(io.StringIO()):
                result=audit.run(ref,root/'audit',baseline_csv=base,expected_names=['a.jpeg','empty.jpeg'])
            self.assertEqual(result['images'],2)
            self.assertFalse(result['raw_trace_available'])
            self.assertEqual(result['categories'],{'weak_final_overlap_raw_unavailable':1})
            self.assertEqual(before,{p:p.read_bytes() for p in before})
            with self.assertRaisesRegex(ValueError,'new audit'):
                audit.run(ref,root/'audit',baseline_csv=base,expected_names=['a.jpeg'])

    def test_cache_mode_lineage_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'inference'
            _,_,_,mask=cache_fixture(cache);reference=root/'reference.csv';csv_file(reference,{'a':[mask]})
            before={p:audit.file_sha(p) for p in cache.rglob('*') if p.is_file()}
            with patch('public_reference_stage_trace.replay_stages',return_value={'baseline_final':[mask]}), redirect_stdout(io.StringIO()):
                result=audit.run(reference,root/'audit',cache_root=cache)
            self.assertEqual(result['mode'],'completed_inference_cache')
            self.assertEqual(result['categories'],{'similar_final_instance':1})
            self.assertEqual(before,{p:audit.file_sha(p) for p in before})
            with self.assertRaisesRegex(ValueError,'separate'):
                audit.run(reference,cache/'audit',cache_root=cache)

    def test_missing_stage_does_not_become_negative_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'cache';image,identity,runtime,_=cache_fixture(root)
            (root/'cache/a/tiled.json.gz').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete cache'):
                audit.load_image(root,image,identity,runtime,{})

    def test_checksum_and_dependency_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'cache';image,identity,runtime,_=cache_fixture(root)
            path=root/'cache/a/final.json.gz'
            envelope=json.loads(gzip.decompress(path.read_bytes()))
            envelope['payload']['data']['baseline_sha256']='wrong'
            path.write_bytes(gzip.compress(audit.canonical(envelope)))
            with self.assertRaisesRegex(ValueError,'checksum'):
                audit.load_image(root,image,identity,runtime,{})
            envelope['sha256']=audit.fingerprint(envelope['payload'])
            path.write_bytes(gzip.compress(audit.canonical(envelope)))
            with self.assertRaisesRegex(ValueError,'dependency'):
                audit.load_image(root,image,identity,runtime,{})


if __name__=='__main__':unittest.main()
