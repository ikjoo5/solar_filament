"""Native-shape, identical-start and learning checks for one model change."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import segmentation_models_pytorch as smp

TOOLS=Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
from paper_multiscale import (MultiScaleResidualContext,PaperUNet,
                             build_from_native_checkpoint,load_base_weights)


class PaperMultiscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.original=smp.Unet(encoder_name='tu-convnext_tiny',encoder_weights=None,
                              in_channels=3,classes=2,activation=None)

    def test_control_and_zero_gate_exactly_match_original_native_output(self):
        torch.manual_seed(7)
        image=torch.randn(1,3,128,160)
        base=copy.deepcopy(self.original).eval()
        control=PaperUNet(copy.deepcopy(base),'control').eval()
        context=PaperUNet(copy.deepcopy(base),'multiscale_context').eval()
        with torch.no_grad():
            expected=base(image)
            self.assertTrue(torch.equal(control(image),expected))
            self.assertTrue(torch.equal(context(image),expected))
        self.assertEqual(tuple(expected.shape),(1,2,128,160))
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            expected=base(image)
            self.assertTrue(torch.equal(control(image),expected))
            self.assertTrue(torch.equal(context(image),expected))

    def test_adapter_construction_preserves_rng_and_adds_under_two_percent(self):
        base=copy.deepcopy(self.original)
        state=torch.get_rng_state().clone()
        model=PaperUNet(base,'multiscale_context')
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        added=sum(p.numel() for p in model.context.parameters())
        count=sum(p.numel() for p in base.parameters())
        self.assertEqual(added,512641)
        self.assertLess(added/count,.02)
        self.assertFalse(any(isinstance(m,(nn.BatchNorm2d,nn.Dropout,nn.Dropout2d)) for m in model.context.modules()))

    def test_gate_gradient_then_branch_learning_and_batch1_global_pool(self):
        torch.manual_seed(10)
        model=MultiScaleResidualContext(32,hidden=16)
        image=torch.randn(1,32,4,5)
        result=model(image)
        self.assertTrue(torch.equal(result,image))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(model.gate.grad))
        self.assertGreater(abs(float(model.gate.grad)),0)
        self.assertTrue(all(p.grad is None or not torch.count_nonzero(p.grad) for n,p in model.named_parameters() if n!='gate'))
        model.zero_grad(set_to_none=True)
        with torch.no_grad():model.gate.fill_(.1)
        model(image).square().mean().backward()
        self.assertTrue(any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in model.named_parameters() if n!='gate'))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_strict_warmstart_and_complete_adapter_resume(self):
        checkpoint=dict(config={'encoder':'tu-convnext_tiny'},model_state_dict=self.original.state_dict(),epoch=84)
        model=build_from_native_checkpoint(checkpoint,'multiscale_context')
        for name,value in self.original.state_dict().items():
            self.assertTrue(torch.equal(model.base.state_dict()[name],value),name)
        with torch.no_grad():model.context.gate.fill_(.25)
        target=PaperUNet(copy.deepcopy(self.original),'multiscale_context')
        target.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(float(target.context.gate.detach()),.25)
        with self.assertRaisesRegex(ValueError,'complete PaperUNet resume'):
            load_base_weights(target,model.state_dict())
        broken=dict(self.original.state_dict());broken.pop(next(iter(broken)))
        with self.assertRaises(RuntimeError):load_base_weights(target,broken)
        with self.assertRaisesRegex(ValueError,'encoder differs'):
            build_from_native_checkpoint(checkpoint,encoder='resnet18')

    def test_invalid_method_shapes_and_classification_heads_fail_explicitly(self):
        with self.assertRaises(ValueError):PaperUNet(copy.deepcopy(self.original),'unknown')
        base=copy.deepcopy(self.original);base.classification_head=nn.Identity()
        with self.assertRaisesRegex(ValueError,'segmentation heads only'):PaperUNet(base)
        model=PaperUNet(copy.deepcopy(self.original))
        with self.assertRaises(ValueError):model(torch.randn(1,1,64,64))
        with self.assertRaises(ValueError):model(torch.randn(3,64,64))
        # SMP0.5 U-Net permits odd shapes and restores their native dimensions.
        model.eval()
        with torch.no_grad():self.assertEqual(tuple(model(torch.randn(1,3,65,64)).shape),(1,2,65,64))
        for args in ((1,128,(1,3,5)),(32,129,(1,3,5)),(32,16,(1,1,3)),(32,16,(0,3,5))):
            with self.assertRaises(ValueError):MultiScaleResidualContext(*args)


if __name__=='__main__':unittest.main()
