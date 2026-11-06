"""Last-checkpoint snapshot and resumable inference provenance without a GPU."""
import ast
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('last_export_cell', ROOT/'tools/colab_export_context_last.py')
cell=importlib.util.module_from_spec(spec);spec.loader.exec_module(cell)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value))


class LastExportTests(unittest.TestCase):
    def test_copy_cell_is_current_compilable_and_never_calls_training_or_gt_score(self):
        source=(ROOT/'tools/colab_export_context_last.py').read_text()
        self.assertEqual(source,(ROOT/'notebooks/ContextLastCheckpoint_Export_CopyCell.py').read_text())
        compile(source,'copycell','exec');tree=ast.parse(cell.WORKER)
        forbidden={'run_training','run_suite','run_context','validate_native','backward','step'}
        calls={node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertFalse(forbidden&calls)
        self.assertIn('export_rich',calls)
        self.assertIn("torch.cuda.is_bf16_supported = lambda *args, **kwargs: False",cell.WORKER)

    def test_original_learning_sources_restore_after_runtime_expiry_and_tamper_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'drive_run';run.mkdir();work=root/'new_runtime'
            for name in ('paper_method_training.py','paper_multiscale.py'):
                (run/name).write_bytes((ROOT/'tools'/name).read_bytes())
            (run/'base_runtime.py').write_bytes(b'\n'+(ROOT/'tools/solar_a100_train_standalone.py').read_bytes())
            cell.restore_learning_sources(run,work)
            self.assertEqual({name:digest(work/name) for name in cell.SOURCE_SHA},cell.SOURCE_SHA)
            (run/'paper_multiscale.py').write_text('altered')
            with self.assertRaisesRegex(ValueError,'SHA'):
                cell.restore_learning_sources(run,work)

    def test_snapshot_remains_one_complete_version_when_training_atomically_replaces_last(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'last.pt';replacement=root/'new.pt';target=root/'local/best.pt'
            old=b'old'+b'0'*(9*1024**2);new=b'new'+b'1'*(10*1024**2)
            source.write_bytes(old);replacement.write_bytes(new)
            real_open=Path.open
            class StableHandle:
                def __init__(self,stream):self.stream=stream;self.changed=False
                def __enter__(self):return self
                def __exit__(self,*args):self.stream.close()
                def read(self,n):
                    value=self.stream.read(n)
                    if not self.changed:
                        replacement.replace(source);self.changed=True
                    return value
            def patched_open(path,*args,**kwargs):
                stream=real_open(path,*args,**kwargs)
                return StableHandle(stream) if path==source and args==('rb',) else stream
            with patch.object(Path,'open',patched_open):
                sha=cell.snapshot_checkpoint(source,target)
            self.assertEqual(target.read_bytes(),old);self.assertEqual(source.read_bytes(),new)
            self.assertEqual(sha,hashlib.sha256(old).hexdigest())

    def test_worker_exports_actual_last_epoch_with_resume_records_without_touching_training_weights(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder);work=root/'work';work.mkdir();training=root/'training';training.mkdir()
            baseline=root/'original_best84.pt';baseline.write_bytes(b'original best84 unchanged')
            cfg=dict(paper_method='multiscale_context',amp_dtype='torch.float16',
                initial_checkpoint=str(baseline),initial_checkpoint_sha256=digest(baseline),infer_batch=1,crop=1024,overlap=.25,
                expected_source_sha256={},source_sha256={},research_source_sha256={})
            for key in ('annotations','train_split','val_split','synthetic_annotations','synthetic_split'):
                path=root/(key+'.csv')
                path.write_text('file_name\na.jpeg\nb.jpeg\n' if key=='val_split' else '{}')
                cfg[key]=str(path);cfg['source_sha256'][key]=digest(path);cfg['expected_source_sha256'][key]=digest(path)
            for name in cell.SOURCE_SHA:
                (work/name).write_text('# fixture')
                cfg['research_source_sha256'][name]=digest(work/name)
            last=training/'last.pt';best=training/'best.pt';best.write_bytes(b'selected epoch45 unchanged')
            checkpoint=dict(epoch=65,config=cfg,history=[dict(epoch=65,validation=dict(pq=.44,tp_pairs=490,fp=300,fn=218))],
                best_epoch=45,model_state_dict={'weight':torch.tensor([1.,2.])},optimizer_state_dict={'unused':'must stay unused'})
            torch.save(checkpoint,last);last_sha=digest(last);original_last=last.read_bytes();original_best=best.read_bytes()
            local_run=work/'local_export_run';sha=cell.snapshot_checkpoint(last,local_run/'best.pt')
            old_records=root/'old_records';old_records.mkdir();(old_records/'sentinel').write_text('keep')
            (local_run/'rich_validation').symlink_to(old_records,target_is_directory=True)
            worker=work/'export_last_worker.py';worker.write_text(cell.WORKER)
            runtime=SimpleNamespace(build_model=lambda *a,**k:None)
            def forbidden(*args,**kwargs):raise AssertionError('No training or GT scoring is allowed')
            runtime.run_training=forbidden;runtime.validate_native=forbidden
            calls=[]
            def export(runtime,current,run):
                calls.append((current,run))
                copied=torch.load(run/'best.pt',weights_only=True)
                self.assertEqual(copied['epoch'],65)
                self.assertTrue(torch.equal(copied['model_state_dict']['weight'],checkpoint['model_state_dict']['weight']))
                self.assertTrue((run/'rich_validation').is_symlink())
                expected=root/'drive/paper_method_experiments/context_last_exports'/f'epoch65_{sha[:12]}'/'records'
                self.assertEqual((run/'rich_validation').resolve(),expected)
                (expected/'a.json').write_text('resumable fixture')
                raw=run/'raw_paper_views.jsonl'
                raw.write_text(''.join(json.dumps(dict(file_name=n,checkpoint_sha256=sha))+'\n' for n in ('a.jpeg','b.jpeg')))
                return raw
            legacy=SimpleNamespace(digest=digest,INITIAL_CHECKPOINT_SHA=digest(baseline),
                atomic_json=write_json,load_runtime=lambda source:runtime,export_rich=export,
                run_training=forbidden,run_suite=forbidden)
            stack.enter_context(patch.dict(sys.modules,{'paper_method_training':legacy}))
            stack.enter_context(patch.object(torch.cuda,'is_available',return_value=True))
            stack.enter_context(patch.object(torch.cuda,'get_device_name',return_value='mockT4'))
            stack.enter_context(patch.object(torch.cuda,'is_bf16_supported',return_value=False))
            stack.enter_context(patch.object(nn.Conv2d,'_conv_forward',nn.Conv2d._conv_forward))
            result=root/'result.json'
            stack.enter_context(patch.object(sys,'argv',[str(worker),'--local-run',str(local_run),
                '--checkpoint-sha',sha,'--base',str(root/'drive'),'--training-run',str(training),'--result',str(result)]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            exec(compile(cell.WORKER,str(worker),'exec'),dict(__name__='__main__',__file__=str(worker)))
            answer=json.loads(result.read_text());summary=answer['summary']
            self.assertEqual(summary['epoch'],65);self.assertEqual(summary['previously_selected_best_epoch'],45)
            self.assertFalse(summary['training_performed']);self.assertFalse(summary['gt_scoring_performed'])
            self.assertEqual(summary['checkpoint_sha256'],last_sha)
            self.assertEqual(summary['stored_validation_at_this_epoch']['pq'],.44)
            self.assertEqual(summary['physical_images'],2);self.assertEqual(len(calls),1)
            self.assertEqual(last.read_bytes(),original_last);self.assertEqual(best.read_bytes(),original_best)
            self.assertTrue((old_records/'sentinel').is_file())
            with zipfile.ZipFile(answer['archive']) as archive:
                self.assertEqual(set(archive.namelist()),{'raw_paper_views.jsonl','summary.json','config.json'})


if __name__=='__main__':unittest.main()
