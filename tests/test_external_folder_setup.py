"""Safe, repeatable Drive-layout setup, exercised only in temporary local folders."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError

TOOLS=Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
import colab_setup_external_magfilo as setup


class ExternalSetupTests(unittest.TestCase):
    def test_creates_evaluation_layout_and_never_overwrites_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=setup.setup_external(root, download=False)
            external=root/'external_MAGFiLO'
            for name in ('images','manifests','evaluations'):self.assertTrue((external/name).is_dir())
            role=json.loads((external/'DATA_ROLE.json').read_text());self.assertFalse(role['allow_training'])
            source=b'{"images":[{"id":"a"}],"annotations":[]}'
            (external/'annotations.json').write_bytes(source)
            cfg=external/'manifests/evaluation_config.json';before=cfg.read_bytes()
            again=setup.setup_external(root, download=True, opener=lambda *a,**k:self.fail('Network must not be called'))
            self.assertEqual(again['download_status'],'existing_annotations_preserved')
            self.assertEqual((external/'annotations.json').read_bytes(),source);self.assertEqual(cfg.read_bytes(),before)
            self.assertEqual(result['images_downloaded'],0)

    def test_http403_keeps_created_folders_and_existing_files(self):
        def denied(request,**kwargs):raise HTTPError(request.full_url,403,'Forbidden',None,None)
        with tempfile.TemporaryDirectory() as directory:
            result=setup.setup_external(directory, opener=denied)
            self.assertEqual(result['download_status'],'download_unavailable')
            self.assertIn('403',result['error']);self.assertTrue(Path(result['images']).is_dir())
            self.assertFalse(Path(result['annotations']).exists())

    def test_unambiguous_json_download_is_validated_and_images_not_fetched(self):
        calls=[]
        def opener(request,**kwargs):
            calls.append(request.full_url)
            data=dict(data=dict(latestVersion=dict(files=[dict(dataFile=dict(id=7,filename='MAGFiLO_Annotations.json'))])))
            return io.BytesIO(json.dumps(data).encode() if len(calls)==1 else b'{"images":[{"id":1}],"annotations":[]}')
        with tempfile.TemporaryDirectory() as directory:
            result=setup.setup_external(directory, opener=opener)
            self.assertEqual(result['download_status'],'annotations_downloaded');self.assertEqual(len(calls),2)
            self.assertTrue(calls[-1].endswith('/access/datafile/7'));self.assertEqual(result['images_downloaded'],0)

    def test_ambiguous_files_and_invalid_download_do_not_create_annotation_file(self):
        def ambiguous(request,**kwargs):
            return io.BytesIO(json.dumps(dict(data=dict(latestVersion=dict(files=[
                dict(dataFile=dict(id=i,filename=f'Annotations_v{i}.json')) for i in (1,2)])))).encode())
        with tempfile.TemporaryDirectory() as directory:
            result=setup.setup_external(directory, opener=ambiguous)
            self.assertEqual(result['download_status'],'annotation_file_selection_needed')
            self.assertFalse(Path(result['annotations']).exists())
            invalid=setup.setup_external(directory,file_id=7,opener=lambda *a,**k:io.BytesIO(b'{"not_coco":true}'))
            self.assertEqual(invalid['download_status'],'download_unavailable')
            self.assertFalse(Path(result['annotations']).exists())

    def test_wrong_existing_role_fails_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'external_MAGFiLO';root.mkdir()
            source=b'{"data_role":"training"}';(root/'DATA_ROLE.json').write_bytes(source)
            with self.assertRaises(ValueError):setup.setup_external(directory,download=False)
            self.assertEqual((root/'DATA_ROLE.json').read_bytes(),source)


if __name__=='__main__':unittest.main()
