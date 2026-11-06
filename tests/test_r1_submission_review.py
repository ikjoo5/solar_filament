"""CPU review of raw images -> frozen full YOLO/best84 -> R1 once.

Synthetic predictions exercise the actual frozen finalize_frame; this does not
validate GPU numerical behavior, trained-model accuracy or a leaderboard score.
"""
from contextlib import redirect_stdout
import copy
import csv
from dataclasses import fields
from functools import lru_cache
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from pycocotools import mask as mu

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import r1_test_submission as runner
import submission_driver as frozen_driver

OUTPUTS = ('submission_pipeline_baseline.csv', 'submission_pipeline_R1.csv', 'results.zip')
VIEWS = ('base', 'scale', 'hflip', 'vflip')
PREPROCESSING = ('limb', 'clahe16', 'clahe32', 'percentile', 'unsharp', 'gaussian', 'bilateral', 'blackhat')


def encode(pixels):
    value = mu.encode(np.asfortranarray(pixels.astype(np.uint8)))
    return dict(size=value['size'], counts=value['counts'].decode('ascii'))


def rectangle(top, left):
    pixels = np.zeros((2048, 2048), np.uint8)
    pixels[top:top+20, left:left+50] = 1
    return pixels


def fixture(root):
    images = root/'images'; images.mkdir()
    names = ['a.png', 'b.png', 'blank.png']
    pixels = np.full((2048, 2048), 100, np.uint8)
    for name in names:
        if not cv2.imwrite(str(images/name), pixels):
            raise RuntimeError('fixture image write failed')
    (root/'names.json').write_text(json.dumps(names))
    (root/'yolo.pt').write_bytes(b'CPU fixture YOLO; not a real checkpoint')
    (root/'native.pt').write_bytes(b'CPU fixture native; not a real checkpoint')
    # No baseline/submission CSV exists. All masks come from inference stages.
    masks = {'a': [rectangle(100, 100), rectangle(350, 350)],
             'b': [rectangle(600, 600)], 'blank': []}
    config = dict(output_root=str(root/'output'), images=str(images),
        expected_names_path=str(root/'names.json'), yolo_checkpoint=str(root/'yolo.pt'),
        native_checkpoint=str(root/'native.pt'), device='cpu', native_batch=1,
        precision='fp16', expected_images=3)
    contract = runner.Contract(images=3, yolo_sha=runner.sha(root/'yolo.pt'),
                               native_sha=runner.sha(root/'native.pt'))
    return config, contract, masks


def peers(mask):
    ys, xs = np.nonzero(mask); top, left = int(ys.min()), int(xs.min())
    first = mask.copy(); first[top+8:top+12, left+50:left+70] = 1
    second = first.copy(); second[top+7, left+50] = 1
    return first, second


@lru_cache(maxsize=16)
def native_template(stem, encoded_masks):
    """Decode real full native records once per synthetic head configuration."""
    heads = np.zeros((2, 2048, 2048), np.float32)
    masks = [mu.decode(dict(size=[2048, 2048], counts=r.encode('ascii'))) for r in encoded_masks]
    if masks:
        heads[0, peers(masks[0])[0] != 0] = .4
        for mask in masks:
            heads[0, mask != 0] = .95; heads[1, mask != 0] = .8
    unions = {view: encode(heads[0] >= .35) for view in ('base', 'hflip', 'vflip', 'hvflip')}
    return runner.native_record_from_heads(heads, heads, unions,
        file_name=stem+'.png', image_sha256='template', identity='template')


class FakeEngine:
    def __init__(self, masks, *, fail_stage=None, mutate=None, malformed_native=False):
        self.masks, self.calls = masks, []
        self.yolo_live = False
        self.fail_stage, self.mutate = fail_stage, mutate
        self.malformed_native = malformed_native

    def start(self, stage, stem):
        self.calls.append((stage, stem))
        if self.fail_stage == stage:
            raise RuntimeError('simulated session interruption: '+stage)

    def view(self, stem, view, values, **extra):
        return dict(image=stem, view=view, shape=[2048, 2048], coordinate_space='original',
            instances=[dict(confidence=confidence, rle=encode(mask)['counts'], **flags)
                       for mask, confidence, flags in values], **extra)

    def geometric(self, stem, pixels):
        self.start('geometric', stem); self.yolo_live = True
        outputs = []
        for view in VIEWS:
            values = []
            for index, original in enumerate(self.masks[stem]):
                mask, confidence = original, .9
                if index == 0 and view in ('scale', 'hflip'):
                    mask = peers(original)[0 if view == 'scale' else 1]; confidence = .7
                values.append((mask, confidence, {}))
            outputs.append(self.view(stem, view, values))
        return outputs

    def preprocessing(self, stem, pixels):
        self.start('preprocessing', stem); self.yolo_live = True
        return [self.view(stem, view, [(mask, .9, {}) for mask in self.masks[stem]])
                for view in PREPROCESSING]

    def tiled(self, stem, pixels):
        self.start('tiled', stem); self.yolo_live = True
        return [self.view(stem, f'tile{top}_{left}_{size}',
                    [(mask, .9, dict(edge_clipped=False)) for mask in self.masks[stem]],
                    crop_bbox=[left, top, 1024, 1024], input_size=size)
                for top in (0, 512, 1024) for left in (0, 512, 1024) for size in (1536, 2048)]

    def finish_geometric(self):
        self.calls.append(('finish_geometric', None)); self.yolo_live = False

    def native(self, stem, pixels, **metadata):
        self.start('native', stem)
        if self.yolo_live:
            raise AssertionError('YOLO was not released before native inference')
        if self.mutate:
            self.mutate(); self.mutate = None
        if self.malformed_native:
            return dict(metadata, size=[2048, 2048], unions={'foreground': {}}, tta={})
        record = copy.deepcopy(native_template(stem, tuple(encode(mask)['counts'] for mask in self.masks[stem])))
        record.update(metadata); record['tta'].update(metadata)
        return record

    def close(self):
        self.calls.append(('close', None))


class R1SubmissionReview(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config, self.contract, self.masks = fixture(self.root)
        self.output = Path(self.config['output_root'])

    def execute(self, engine=None, **kwargs):
        engine = engine or FakeEngine(self.masks)
        with redirect_stdout(io.StringIO()):
            result = runner.run(self.config, contract=self.contract, engine_factory=lambda cfg: engine, **kwargs)
        return result, engine

    def no_export(self):
        for name in OUTPUTS:
            self.assertFalse((self.output/name).exists(), name)

    def rows(self, name):
        with (self.output/name).open(newline='') as stream:
            return list(csv.DictReader(stream))

    def test_full_pipeline_partial_resume_completed_reexport_apply_rule_once(self):
        path_open = Path.open
        atomic_bytes = runner.atomic_bytes
        generated_csvs = set()
        def track_export(path, data):
            result = atomic_bytes(path, data)
            if Path(path).suffix == '.csv':
                generated_csvs.add(Path(path).resolve())
            return result
        def no_input_csv(path, mode='r', *args, **kwargs):
            if path.suffix == '.csv' and ('r' in mode or '+' in mode) and path.resolve() not in generated_csvs:
                raise AssertionError('No submission CSV may be read as an input')
            return path_open(path, mode, *args, **kwargs)
        with patch.object(runner, 'finalize_frame', wraps=frozen_driver.finalize_frame) as finalizer, \
             patch.object(runner, 'restore_consensus_tails', wraps=runner.restore_consensus_tails) as rule:
            with patch.object(Path, 'open', no_input_csv), patch.object(runner, 'atomic_bytes', side_effect=track_export):
                partial, first = self.execute(max_images=1)
            self.assertEqual(partial['status'], 'paused'); self.assertEqual(partial['completed_images'], 1)
            self.assertEqual((finalizer.call_count, rule.call_count), (1, 1)); self.no_export()
            self.assertEqual(first.calls, [(stage, 'a') for stage in runner.INFERENCE_STAGES[:3]]+
                             [('finish_geometric', None), ('native', 'a'), ('close', None)])
            with patch.object(Path, 'open', no_input_csv), patch.object(runner, 'atomic_bytes', side_effect=track_export):
                complete, second = self.execute()
            self.assertEqual(complete['status'], 'complete')
            self.assertEqual((finalizer.call_count, rule.call_count), (3, 3))
            self.assertEqual([call for call in second.calls if call[0] in runner.INFERENCE_STAGES],
                [(stage, stem) for stem in ('b', 'blank') for stage in runner.INFERENCE_STAGES[:3]]+
                [('native', 'b'), ('native', 'blank')])
            for call in finalizer.call_args_list:
                stages = call.args[1]
                self.assertEqual(set(stages), set(runner.INFERENCE_STAGES))
                self.assertEqual([len(stages[key]) for key in runner.INFERENCE_STAGES[:3]], [4, 8, 18])
                self.assertIn('spine', stages['native']['unions'])
                self.assertEqual(set(stages['native']['components']), {'0.2', '0.35', '0.5', '0.65', '0.8'})
            expected = (self.output/OUTPUTS[1]).read_bytes()
            (self.output/OUTPUTS[1]).write_bytes(b'corrupted export; immutable caches retained')
            with patch.object(runner, 'device_facts', side_effect=AssertionError('no GPU query')):
                with redirect_stdout(io.StringIO()):
                    result = runner.run(self.config, contract=self.contract,
                        engine_factory=lambda cfg: self.fail('completed cache created engine'))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual((finalizer.call_count, rule.call_count), (3, 3))
            self.assertEqual(expected, (self.output/OUTPUTS[1]).read_bytes())
        before, after = self.rows(OUTPUTS[0]), self.rows(OUTPUTS[1])
        self.assertEqual([row['filament_id'] for row in before], ['a_1', 'a_2', 'b_1'])
        self.assertEqual([row['filament_id'] for row in before], [row['filament_id'] for row in after])
        self.assertEqual(complete['mask_count'], 3)
        self.assertEqual(complete['changed_masks'], 2); self.assertEqual(complete['added_pixels'], 160)
        for old, new, original in zip(before, after, self.masks['a']+self.masks['b']):
            self.assertEqual(old['segmentation_rle'], encode(original)['counts'])
            decoded = mu.decode(dict(size=[2048, 2048], counts=new['segmentation_rle'].encode()))
            self.assertTrue(np.all(decoded[original != 0])); self.assertLessEqual(decoded.sum(), original.sum()*1.25)
        manifest = json.loads((self.output/'run_manifest.json').read_text())
        self.assertEqual(manifest['empty_baseline_images'], 1); self.assertEqual(manifest['anchored_images'], 2)
        self.assertIsNone(manifest['new_public_LB']); self.assertIsNone(manifest['current_baseline_public_LB'])
        self.assertFalse(list(self.output.rglob('*.pending')))

    def test_blank_image_gets_all_stages_and_count_is_dynamic(self):
        self.masks['b'].append(rectangle(800, 800))
        result, engine = self.execute()
        self.assertEqual(result['mask_count'], 4)
        self.assertEqual([call[0] for call in engine.calls if call[1] == 'blank'], list(runner.INFERENCE_STAGES))
        self.assertFalse(any(row['filament_id'].startswith('blank_') for row in self.rows(OUTPUTS[1])))
        final = json.loads(gzip.decompress((self.output/'cache/blank/final.json.gz').read_bytes()))
        self.assertEqual(final['payload']['data']['masks'], [])

    def test_interruption_reuses_all_yolo_stages_and_phase_order(self):
        failed = FakeEngine(self.masks, fail_stage='native')
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.execute(failed)
        self.no_export()
        expected = [(stage, stem) for stem in ('a', 'b', 'blank') for stage in runner.INFERENCE_STAGES[:3]]
        self.assertEqual(failed.calls, expected+[('finish_geometric', None), ('native', 'a'), ('close', None)])
        for stem in ('a', 'b', 'blank'):
            for stage in runner.INFERENCE_STAGES[:3]:
                self.assertTrue((self.output/f'cache/{stem}/{stage}.json.gz').is_file())
        _, resumed = self.execute()
        self.assertFalse(any(call[0] in runner.INFERENCE_STAGES[:3] for call in resumed.calls))
        self.assertEqual([call for call in resumed.calls if call[0] == 'native'],
                         [('native', 'a'), ('native', 'b'), ('native', 'blank')])

    def test_interruption_inside_yolo_reuses_only_finished_stage(self):
        with self.assertRaisesRegex(RuntimeError, 'preprocessing'):
            self.execute(FakeEngine(self.masks, fail_stage='preprocessing'), max_images=1)
        self.assertTrue((self.output/'cache/a/geometric.json.gz').is_file())
        self.assertFalse((self.output/'cache/a/preprocessing.json.gz').exists())
        _, engine = self.execute(max_images=1)
        self.assertEqual(engine.calls, [('preprocessing', 'a'), ('tiled', 'a'), ('finish_geometric', None),
                                        ('native', 'a'), ('close', None)])
        self.no_export()

    def test_native_and_baseline_saved_before_rule_interruption_are_reused(self):
        with patch.object(runner, 'restore_consensus_tails', side_effect=RuntimeError('after baseline save')):
            with self.assertRaisesRegex(RuntimeError, 'after baseline save'):
                self.execute(max_images=1)
        for stage in (*runner.INFERENCE_STAGES, 'baseline'):
            self.assertTrue((self.output/f'cache/a/{stage}.json.gz').is_file())
        with patch.object(runner, 'finalize_frame', side_effect=AssertionError('baseline must be reused')):
            with redirect_stdout(io.StringIO()):
                result = runner.run(self.config, max_images=1, contract=self.contract,
                    engine_factory=lambda cfg: self.fail('complete support caches should avoid engine'))
        self.assertEqual(result['completed_images'], 1); self.no_export()

    def test_legacy_submission_input_and_wrong_weights_rejected_before_engine(self):
        for key in ('baseline_csv', 'baseline_sha', 'masks', 'anchored_images'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'no unknown keys'):
                self.execute_config(dict(self.config, **{key: 'forbidden'}))
        for key in ('yolo_checkpoint', 'native_checkpoint'):
            with self.subTest(key=key):
                path = Path(self.config[key]); original = path.read_bytes()
                path.write_bytes(original+b'wrong checkpoint')
                with self.assertRaisesRegex(ValueError, 'Wrong frozen input SHA-256'):
                    self.execute_config(self.config)
                path.write_bytes(original)
        self.no_export()

    def execute_config(self, config):
        with redirect_stdout(io.StringIO()):
            return runner.run(config, contract=self.contract,
                engine_factory=lambda cfg: self.fail('invalid inputs created an engine'))

    def test_input_mutation_during_native_prevents_export(self):
        for key in ('yolo_checkpoint', 'native_checkpoint', 'images'):
            with self.subTest(key=key):
                self.config['output_root'] = str(self.root/('mutated_'+key))
                target = Path(self.config['images'])/'blank.png' if key == 'images' else Path(self.config[key])
                original = target.read_bytes()
                try:
                    with self.assertRaisesRegex(ValueError, 'changed during the run|changed after validation'):
                        self.execute(FakeEngine(self.masks, mutate=lambda: target.write_bytes(original+b'changed')))
                    self.assertFalse((Path(self.config['output_root'])/OUTPUTS[1]).exists())
                finally:
                    target.write_bytes(original)

    def test_exact_file_membership_and_resume_input_identity(self):
        extra = self.root/'images/extra.png'; extra.write_bytes((self.root/'images/a.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            self.execute()
        extra.unlink(); self.execute(max_images=1)
        blank = self.root/'images/blank.png'; blank.write_bytes(blank.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'Inputs/config/code/runtime changed'):
            self.execute()
        self.no_export()

    def test_cache_checksum_support_chain_and_missing_stage_fail_closed(self):
        self.execute(max_images=1)
        path = self.output/'cache/a/preprocessing.json.gz'; original = path.read_bytes()
        envelope = json.loads(gzip.decompress(original)); envelope['payload']['data'][0]['instances'][0]['confidence'] = .8
        path.write_bytes(gzip.compress(json.dumps(envelope).encode()))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.execute()
        runner.write_cache(path, envelope['payload'])
        with self.assertRaisesRegex(ValueError, 'no longer matches its four inference stages'):
            self.execute()
        path.write_bytes(original)
        envelope = json.loads(gzip.decompress(original)); envelope['payload']['identity'] = 'different run'
        runner.write_cache(path, envelope['payload'])
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.execute()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing one of its full-pipeline inference stages'):
            self.execute()
        self.no_export()

    def test_native_requires_full_rich_fields(self):
        with self.assertRaisesRegex(ValueError, 'full native rich record'):
            self.execute(FakeEngine(self.masks, malformed_native=True), max_images=1)
        self.assertFalse((self.output/'cache/a/native.json.gz').exists()); self.no_export()

    def test_runtime_binding_tamper_and_new_inference_runtime_mismatch(self):
        self.execute(max_images=1)
        path = self.output/'runtime_device.json'; original = path.read_bytes()
        runtime = json.loads(original); runtime['payload']['facts']['device'] = 'other device'
        path.write_text(json.dumps(runtime))
        with self.assertRaisesRegex(ValueError, 'Runtime device snapshot checksum'):
            self.execute()
        path.write_bytes(original)
        with patch.object(runner, 'device_facts', return_value={'different_gpu': 'A100'}):
            with self.assertRaisesRegex(ValueError, 'CUDA/GPU/cuDNN runtime changed'):
                self.execute()
        self.no_export()

    def test_single_output_process_lock_and_release(self):
        key = hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()
        locks = Path(tempfile.gettempdir())/'solar_r1_locks'; locks.mkdir(exist_ok=True)
        with (locks/(key+'.lock')).open('a') as held:
            runner.fcntl.flock(held, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'already using this output_root'):
                self.execute()
            runner.fcntl.flock(held, runner.fcntl.LOCK_UN)
        result, _ = self.execute(max_images=1)
        self.assertEqual(result['status'], 'paused')

    def test_nonproduction_contract_requires_injection_and_production_constants(self):
        with self.assertRaisesRegex(ValueError, 'require an injected'):
            runner.run(self.config, contract=self.contract)
        production = runner.Contract()
        self.assertEqual(production.images, 180)
        self.assertEqual({field.name for field in fields(production)}, {'images', 'yolo_sha', 'native_sha'})
        import submission_inference as frozen
        self.assertEqual(production.yolo_sha, frozen.YOLO_SHA256)
        self.assertEqual(production.native_sha, frozen.NATIVE_SHA256)


class FrozenInferenceReview(unittest.TestCase):
    def test_support_threshold_matches_frozen_record_base_and_tta(self):
        import submission_inference as frozen
        base = np.zeros((2, 2048, 2048), np.float32)
        average = np.zeros_like(base)
        base[0, 100:104, 100:104] = np.float32(.35)
        base[0, 200:204, 200:204] = np.nextafter(np.float32(.35), np.float32(0))
        average[0, 300:304, 300:304] = .8
        unions = {view: encode(base[0] >= .35) for view in ('base', 'hflip', 'vflip', 'hvflip')}
        result = runner.native_record_from_heads(base, average, unions, file_name='a.png', image_sha256='s', identity='id')
        for actual, heads in ((result, base), (result['tta'], average)):
            expected = frozen.prediction_record(heads, 'a.png', 's', 'id')
            for key, value in expected.items():
                self.assertEqual(actual[key], value)
            self.assertEqual(actual['unions']['foreground']['0.35'], expected['unions']['foreground']['0.35'])
            self.assertEqual(int(mu.area(runner.union_rle(actual['unions']['foreground']['0.35']))), 16)

    def test_four_geometric_views_reverse_flips_in_original_coordinates(self):
        import submission_inference as frozen
        import torch
        provider = frozen.InferenceProvider(None, None, device='cpu')
        pixels = np.zeros((2048, 2048), np.uint8)
        pixels[75:90, 100:140] = 1
        calls = []
        def result(image, size):
            calls.append((image.copy(), size))
            return SimpleNamespace(masks=SimpleNamespace(data=torch.from_numpy(image[None].copy())),
                                   boxes=SimpleNamespace(conf=torch.tensor([.8])))
        with patch.object(frozen, 'limb', side_effect=lambda image: image), patch.object(provider, '_result', side_effect=result):
            views = provider.geometric('a', pixels)
        runner.validate_geometry(views, 'a')
        self.assertEqual([size for _, size in calls], [2048, 2560, 2048, 2048])
        self.assertTrue(np.array_equal(calls[2][0], np.fliplr(pixels)))
        self.assertTrue(np.array_equal(calls[3][0], np.flipud(pixels)))
        for view in views:
            self.assertEqual(view['instances'][0]['rle'], encode(pixels)['counts'])

    def test_native_worker_calls_frozen_predict_views_with_fp16_batch1(self):
        import submission_inference as frozen
        import submission_native_process as process
        import native_instance_training as training
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.full((2048, 2048), 10, np.uint8)
            np.save(root/'image.npy', pixels)
            request = dict(protocol=process.PROTOCOL, command='native', id='request', input_path=str(root/'image.npy'),
                           output_path=str(root/'output.json'), file_name='a.png', image_sha256='sha', identity='id')
            stdin = io.StringIO(json.dumps(request)+'\n'+json.dumps(dict(protocol=process.PROTOCOL, command='close'))+'\n')
            stdout = io.StringIO()
            provider = SimpleNamespace(device=torch.device('cuda'), _native=lambda: 'fake model')
            heads = np.zeros((2, 2048, 2048), np.float32)
            with patch.object(frozen, 'InferenceProvider', return_value=provider), \
                 patch.object(training, 'configure_cuda'), patch.object(frozen, 'limb', side_effect=lambda image: image+1), \
                 patch.object(frozen, 'predict_views', return_value=(heads, heads, {view: encode(heads[0]) for view in ('base', 'hflip', 'vflip', 'hvflip')})) as predict, \
                 patch.object(sys, 'stdin', stdin), redirect_stdout(stdout):
                runner.native_support_server(['--native-support-server', '--checkpoint', 'fixture', '--device', 'cuda'])
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'ok')
            self.assertEqual(predict.call_args.kwargs['batch'], 1)
            self.assertIs(predict.call_args.kwargs['amp_dtype'], torch.float16)
            np.testing.assert_array_equal(predict.call_args.args[1], pixels+1)
            record = json.loads((root/'output.json').read_text())
            runner.validate_native(record, dict(name='a.png', stem='a', sha256='sha'), 'id')

    def test_engine_snapshots_child_before_yolo_and_releases_before_native(self):
        import submission_inference as frozen
        import submission_native_process as process
        events = []
        class Client:
            def __init__(self, *args, **kwargs):
                events.append('child configuration')
                self.command = kwargs['_command']
            def native(self, *args, **kwargs):
                self_check.assertIsNone(engine.provider)
                events.append('native')
            def close(self):
                events.append('closed')
        class Provider:
            def __init__(self, *args, **kwargs):
                events.append('YOLO provider')
                self._yolo_model = object()
            def geometric(self, *args):
                events.append('YOLO geometric')
            def preprocessing(self, *args):
                events.append('YOLO preprocessing')
            def tiled(self, *args):
                events.append('YOLO tiled')
        self_check = self
        with patch.object(process, 'NativeProcessClient', Client), patch.object(frozen, 'InferenceProvider', Provider):
            engine = runner.SupportEngine(dict(native_checkpoint='native', yolo_checkpoint='yolo', device='cpu'))
            self.assertIn('--native-support-server', engine.client.command)
            for stage in runner.INFERENCE_STAGES[:3]:
                getattr(engine, stage)('a', None)
            engine.finish_geometric()
            engine.native('a', None)
            engine.close()
        self.assertEqual(events, ['child configuration', 'YOLO provider', 'YOLO geometric', 'YOLO preprocessing', 'YOLO tiled', 'native', 'closed'])


if __name__ == '__main__':
    unittest.main()
