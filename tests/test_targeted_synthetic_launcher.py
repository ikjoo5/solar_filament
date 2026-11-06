"""Bounded phase/epoch orchestration checks without launching real training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parents[1]/'tools'
spec=importlib.util.spec_from_file_location('targeted_launch_test',TOOLS/'targeted_synthetic_launcher.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)

class LauncherContract(unittest.TestCase):
    def exercise(self,phase='all',cap=30,arms=launch.ARMS,stages=None):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(output_root=tmp,epochs=30)
            prepared=dict(identity='data',real_samples=1053,synthetic_samples=263,synthetic_fraction=263/1316,
                          arms={'control_v2':{'yaml':'control.yaml'},'targeted_v3':{'yaml':'targeted.yaml'}})
            calls=[]
            def train(c,a,y,i,until):
                calls.append((a,until,y,i));return dict(checkpoint=a+'.pt',epochs_completed=until)
            if stages is None:stages={a:dict(checkpoint=a+'.pt',epochs_completed=cap) for a in arms}
            trainer=types.SimpleNamespace(train_arm=train,
                evaluate_checkpoints=lambda c,s:dict(checkpoints=s,tag=c['evaluation_tag']),
                export_results=lambda c:[Path(tmp)/'results_part01.zip'])
            data=types.SimpleNamespace(prepare_dataset=lambda c:prepared)
            with patch.object(launch,'validate_configuration',return_value={'checked':'true'}),\
                 patch.object(launch,'collect_stages',return_value=stages),\
                 patch.dict(sys.modules,{'targeted_synthetic_training':trainer,'targeted_synthetic_data':data}):
                result=launch.run(cfg,phase,cap,arms)
            self.assertEqual(json.loads((Path(tmp)/'result.json').read_text()),result)
            return result,calls

    def test_default_finishes_true_thirty_for_both_arms(self):
        r,calls=self.exercise()
        self.assertEqual([c[1] for c in calls],[30,30]);self.assertEqual(r['status'],'complete')
        self.assertEqual(r['evaluation']['tag'],'epoch030')

    def test_prepare_does_not_train_or_evaluate(self):
        r,calls=self.exercise('prepare')
        self.assertEqual(calls,[]);self.assertIsNone(r['evaluation']);self.assertEqual(r['status'],'prepared')

    def test_ten_is_explicitly_partial_not_hidden_hundred(self):
        r,calls=self.exercise(cap=10)
        self.assertEqual([c[1] for c in calls],[10,10]);self.assertEqual(r['fixed_final_epoch'],30)
        self.assertEqual(r['status'],'stage_complete_final_training_remaining')

    def test_one_arm_waits_for_same_stage_other_arm(self):
        r,calls=self.exercise(arms=('control_v2',))
        self.assertEqual(len(calls),1);self.assertIsNone(r['evaluation'])
        self.assertEqual(r['status'],'waiting_for_other_arm')

    def test_evaluation_can_run_without_disposable_dataset(self):
        r,calls=self.exercise('evaluate')
        self.assertEqual(calls,[]);self.assertEqual(r['status'],'complete')

    def test_evaluation_rejects_unpaired_checkpoints(self):
        with self.assertRaises(FileNotFoundError):self.exercise('evaluate',stages={})

    def test_schedule_and_batch_guard_before_io(self):
        c=dict(epochs=30,batch=1,nbs=1,imgsz=2048)
        with self.assertRaises(ValueError):launch.validate_configuration(c,'all',31,launch.ARMS)
        with self.assertRaises(ValueError):launch.validate_configuration(dict(c,nbs=4),'all',30,launch.ARMS)

if __name__=='__main__':unittest.main()
