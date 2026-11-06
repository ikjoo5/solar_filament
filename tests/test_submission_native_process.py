"""Native model isolation via pipe/file IPC; no GPU or model download needed."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from submission_native_process import NativeProcessClient


FAKE_WORKER = r'''
import json,os,sys
from pathlib import Path
import numpy as np
mode=sys.argv[1]
calls=0
print('Loading native model diagnostics',flush=True)
print(json.dumps({'diagnostic':'not a protocol response'}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if request['command']=='close':break
    if mode=='eof':sys.exit(7)
    calls+=1
    if mode=='error':
        print(json.dumps(dict(protocol='solar-native-v1',status='error',id=request['id'],
            error_type='FixtureError',message='failed native request')),flush=True)
        continue
    pixels=np.load(request['input_path'],allow_pickle=False)
    copied=Path(request['output_path']).with_suffix('.roundtrip.npy')
    np.save(copied,pixels,allow_pickle=False)
    record=dict(pid=os.getpid(),calls=calls,shape=list(pixels.shape),dtype=str(pixels.dtype),
        roundtrip=str(copied),environment=os.environ.get('NATIVE_SNAPSHOT'),
        cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        file_name=request['file_name'],image_sha256=request['image_sha256'],identity=request['identity'])
    Path(request['output_path']).write_text(json.dumps(record))
    print(json.dumps(dict(protocol='solar-native-v1',status='ok',
        id=request['id']+(1 if mode=='wrong-id' else 0),output_path=request['output_path'])),flush=True)
'''


def command(mode='normal'):
    return [sys.executable,'-u','-c',FAKE_WORKER,mode]


class NativeProcessTests(unittest.TestCase):
    def test_snapshot_precedes_yolo_mutation_and_process_is_persistent(self):
        pixels=np.random.default_rng(21).integers(0,256,(2048,2048),np.uint8)
        with patch.dict(os.environ,{'NATIVE_SNAPSHOT':'before-yolo','CUBLAS_WORKSPACE_CONFIG':':16:8'}):
            with NativeProcessClient('unused',_command=command()) as client:
                self.assertIsNone(client._process)
                os.environ['NATIVE_SNAPSHOT']='after-yolo'
                os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                first=client.native('first',pixels,file_name='first.png',image_sha256='pixels',identity='job')
                self.assertEqual(first['environment'],'before-yolo')
                self.assertEqual(first['cublas'],':16:8')
                self.assertEqual(first['shape'],[2048,2048]);self.assertEqual(first['dtype'],'uint8')
                self.assertEqual(first['file_name'],'first.png');self.assertEqual(first['identity'],'job')
                np.testing.assert_array_equal(np.load(first['roundtrip'],allow_pickle=False),pixels)
                second=client.native('second',pixels[::-1].copy(),identity='job')
                np.testing.assert_array_equal(np.load(second['roundtrip'],allow_pickle=False),pixels[::-1])
                self.assertEqual(first['pid'],second['pid'])
                self.assertEqual((first['calls'],second['calls']),(1,2))
                directory=Path(client._temporary.name);process=client._process
            self.assertFalse(directory.exists())
            self.assertIsNotNone(process.returncode)
            client.close()  # Idempotent; never leaves a worker behind.

    def test_original_absent_cublas_environment_is_not_inherited_later(self):
        original=dict(os.environ)
        original.pop('CUBLAS_WORKSPACE_CONFIG',None)
        with patch.dict(os.environ,original,clear=True):
            with NativeProcessClient('unused',_command=command()) as client:
                os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                record=client.native('frame',np.zeros((2048,2048),np.uint8))
                self.assertIsNone(record['cublas'])

    def test_worker_error_and_eof_are_propagated_and_reaped(self):
        pixels=np.zeros((2048,2048),np.uint8)
        for mode,expected in [('error','FixtureError: failed native request'),('eof','ended without a response'),
                              ('wrong-id','response ID differs')]:
            client=NativeProcessClient('unused',_command=command(mode))
            with self.assertRaisesRegex(RuntimeError,expected):
                client.native('frame',pixels)
            self.assertTrue(client._closed)
            self.assertIsNone(client._temporary)
            self.assertIsNotNone(client._process.returncode)
            if mode=='eof':self.assertEqual(client._process.returncode,7)

    def test_invalid_pixels_do_not_launch_a_worker(self):
        with NativeProcessClient('unused',_command=command()) as client:
            for pixels in [np.zeros((2048,2048),np.float32),np.zeros((1024,1024),np.uint8)]:
                with self.assertRaises(ValueError):client.native('frame',pixels)
            self.assertIsNone(client._process)

    def test_client_import_and_constructor_do_not_import_torch(self):
        code=(f"import sys;sys.path.insert(0,{str(ROOT/'tools')!r});"
              "from submission_native_process import NativeProcessClient;"
              "client=NativeProcessClient('unused');"
              "assert 'torch' not in sys.modules;assert 'ultralytics' not in sys.modules;"
              "assert client._process is None;client.close();print('isolated')")
        output=subprocess.check_output([sys.executable,'-c',code],text=True)
        self.assertEqual(output.strip(),'isolated')

    def test_actual_server_protocol_constructs_one_provider_for_two_requests(self):
        import submission_inference
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pixels=np.arange(2048**2,dtype=np.uint8).reshape(2048,2048)
            source=root/'pixels.npy';np.save(source,pixels,allow_pickle=False)
            calls=[]
            class FakeProvider:
                def __init__(self,*args,**kw):calls.append(('init',args,kw))
                def native(self,image_id,image,**metadata):
                    np.testing.assert_array_equal(image,pixels)
                    calls.append(('native',image_id,metadata))
                    return dict(image_id=image_id,**metadata)
            requests=[]
            for index in (1,2):
                requests.append(dict(protocol='solar-native-v1',command='native',id=index,
                    image_id=f'frame{index}',input_path=str(source),output_path=str(root/f'{index}.json'),
                    file_name=f'frame{index}.jpeg',image_sha256='pixels',identity='job'))
            requests.append(dict(protocol='solar-native-v1',command='close'))
            stdin=io.StringIO(''.join(json.dumps(r)+'\n' for r in requests));stdout=io.StringIO()
            with patch.object(submission_inference,'InferenceProvider',FakeProvider),patch('sys.stdin',stdin),redirect_stdout(stdout):
                submission_inference.native_server_main(['--native-server','--checkpoint','unused','--device','cpu','--batch','4'])
            responses=[json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual([r['status'] for r in responses],['ok','ok'])
            self.assertEqual([r['id'] for r in responses],[1,2])
            self.assertEqual([c[0] for c in calls],['init','native','native'])
            self.assertEqual(json.loads((root/'2.json').read_text())['file_name'],'frame2.jpeg')


if __name__=='__main__':
    unittest.main()
