"""Read-only discovery of the exact reviewed epoch82 checkpoint on Drive.

A search result is accepted only by its recorded SHA-256, never by a filename.
This helper uses the standard library and does not load checkpoint pickle data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid

CHECKPOINT_SHA = '8ea1198734dd434c42889aee2c79928afd8eb8ce4491e826c13d07e41e03d330'
PRUNED = frozenset({'train_images','test_images','images','predictions','cache','caches',
                    '__pycache__','synthesis','code','node_modules'})
COMMON_BEST = frozenset({'best.pt','best.pth','best_model.pt','best_model.pth'})


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def state_summary(training_root):
    path=Path(training_root)/'train/state.json'
    try:
        value=json.loads(path.read_text());state=value.get('payload',{})
        return dict(path=str(path),completed_epoch=state.get('epoch'),
            best_epoch=state.get('best_epoch'),best_pq=state.get('best_pq'),best=state.get('best'))
    except (OSError,ValueError,AttributeError) as error:
        return dict(path=str(path),read_error=str(error))


def save_report(path,report):
    if path is None:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def find_reviewed_checkpoint(search_root,training_root,checkpoint=None,report_path=None,
                             max_directories=1500,max_candidates=100):
    root=Path(search_root).resolve()
    report=dict(expected_epoch=82,expected_sha256=CHECKPOINT_SHA,search_root=str(root),
        training_state=state_summary(training_root),checked=[],weight_files=[],
        pruned_directory_names=sorted(PRUNED),directories_scanned=0,scan_errors=[])
    print('Current training state:',json.dumps(report['training_state']),flush=True)
    print('Read-only epoch82 checkpoint search:',root,flush=True)
    checked=set()
    def check(path,explicit=False):
        path=Path(path)
        if path.is_symlink() or (not explicit and not path.resolve().is_relative_to(root)):
            if explicit:raise ValueError('Explicit checkpoint must be a regular file, not a symlink')
            return None
        path=path.resolve()
        if path in checked or not path.is_file():return None
        if len(checked)>=max_candidates:
            raise RuntimeError('Checkpoint candidate limit reached; narrow SEARCH_ROOT and rerun')
        checked.add(path)
        print('Checking checkpoint:',path,flush=True)
        actual=digest(path);report['checked'].append(dict(path=str(path),sha256=actual))
        if actual==CHECKPOINT_SHA:
            report.update(status='found',checkpoint=str(path),checkpoint_sha256=actual)
            save_report(report_path,report)
            print('Verified reviewed epoch82 checkpoint:',path,flush=True)
            return path
        print('Different checkpoint bytes; preserved:',path,flush=True)
        return None
    try:
        if checkpoint is not None:
            found=check(checkpoint,explicit=True)
            if found is None:raise ValueError('Explicit checkpoint is missing or does not match reviewed epoch82 SHA-256')
            return found
        if not root.is_dir():raise FileNotFoundError('Checkpoint search root missing: '+str(root))
        # The common original location can complete recovery without a tree scan.
        found=check(Path(training_root)/'train/checkpoints/best_epoch082.pt')
        if found:return found
        fallback=[]
        def onerror(error):report['scan_errors'].append(str(error))
        for directory,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
            directory=Path(directory);report['directories_scanned']+=1
            if report['directories_scanned']>max_directories:
                raise RuntimeError('Checkpoint directory limit reached; narrow SEARCH_ROOT and rerun')
            dirs[:]=sorted(d for d in dirs if not d.startswith('.') and d not in PRUNED
                           and not (directory/d).is_symlink())
            if report['directories_scanned']%100==0:
                print('Checkpoint search directories:',report['directories_scanned'],flush=True)
            for name in sorted(files):
                path=directory/name
                if path.suffix.lower() in ('.pt','.pth'):
                    if len(report['weight_files'])<40:report['weight_files'].append(str(path))
                    if re.search(r'(?<!\d)0?82(?!\d)',path.stem):
                        found=check(path)
                        if found:return found
                    elif name.lower() in COMMON_BEST:fallback.append(path)
                elif name=='state.json' and not path.is_symlink():
                    try:
                        if path.stat().st_size>=1_000_000:continue
                        state=json.loads(path.read_text()).get('payload',{})
                        best=state.get('best',{})
                        if best.get('sha256')==CHECKPOINT_SHA and isinstance(best.get('path'),str):
                            found=check(directory/best['path'])
                            if found:return found
                    except (OSError,ValueError,AttributeError,TypeError):pass
        for candidate in fallback:
            found=check(candidate)
            if found:return found
        report['status']='not_found';save_report(report_path,report)
        print('Checkpoint search report:',json.dumps(report,indent=2),flush=True)
        raise FileNotFoundError('Reviewed epoch82 weights were not found in the searched project. '
            'The current state selects epoch '+str(report['training_state'].get('best_epoch'))+'. '
            'No alternate model was selected. Check the Google account, the completed-run folder, '
            'Drive trash or backup, then set SEARCH_ROOT or CHECKPOINT to that location. '
            'The result ZIP contains metrics, not the missing model weights.')
    except BaseException:
        if report.get('status')!='found':
            report.setdefault('status','search_failed');save_report(report_path,report)
        raise
