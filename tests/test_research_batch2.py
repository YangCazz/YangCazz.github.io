"""Bounded editorial checks; set examples are schematic, not clinical validation."""
import ast
import importlib.util
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOPO = ROOT / '_posts/2026-06-03-topology-segmentation-learning.md'
MAMBA = ROOT / '_posts/2026-06-02-mamba-state-space-model-comprehensive.md'

def score(pred, truth, pred_skeleton, truth_skeleton):
    """Set definition for nonempty schematic skeletons, not a skeleton extractor."""
    precision = len(pred_skeleton & truth) / len(pred_skeleton)
    sensitivity = len(truth_skeleton & pred) / len(truth_skeleton)
    return 0.0 if precision + sensitivity == 0 else 2 * precision * sensitivity / (precision + sensitivity)

class ResearchBatchTests(unittest.TestCase):
    def test_cldice_definition_has_both_directions(self):
        text = TOPO.read_text()
        self.assertIn(r'T_{prec}=\frac{|S_{pred}\cap V_{gt}|}{|S_{pred}|}', text)
        self.assertIn(r'T_{sens}=\frac{|S_{gt}\cap V_{pred}|}{|S_{gt}|}', text)
        self.assertIn(r'\frac{2T_{prec}T_{sens}}{T_{prec}+T_{sens}}', text)
        self.assertIn('Corollary 1.1', text)

    def test_identical_schematic_centerlines(self):
        line = set(range(7))
        self.assertEqual(score(line, line, line, line), 1.0)

    def test_missing_schematic_centerline_segment(self):
        truth = set(range(7)); pred = truth - {3}
        self.assertTrue(math.isclose(score(pred, truth, pred, truth), 12 / 13))

    def test_spurious_schematic_centerline_segment(self):
        truth = set(range(7)); pred = truth | {99}
        self.assertTrue(math.isclose(score(pred, truth, pred, truth), 14 / 15))

    def test_snippet_syntax_and_official_skeleton_delegate(self):
        text = TOPO.read_text().split('## 实战：实现 clDice 损失', 1)[1]
        code = re.search(r'```python\n(.*?)\n```', text, re.S).group(1)
        ast.parse(code)
        self.assertIn('from soft_skeleton import SoftSkeletonize', code)
        self.assertIn('SoftSkeletonize(num_iter=iter_skel)', code)
        self.assertNotIn('torch.where', code)
        self.assertNotIn('soft_topk', code)

    def test_u_mamba_retains_convolution(self):
        text = MAMBA.read_text()
        self.assertNotIn('U-Net 编码器中的卷积块替换为 Mamba 块', text)
        self.assertIn('残差卷积后接 Mamba', text)
        self.assertIn('Enc 和 Bot', text)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('soft_skeleton'),
                         'PyTorch and official SoftSkeletonize are not installed; no numeric/autograd validation claimed')
    def test_optional_torch_wrapper_smoke(self):
        import torch
        text = TOPO.read_text().split('## 实战：实现 clDice 损失', 1)[1]
        code = re.search(r'```python\n(.*?)\n```', text, re.S).group(1)
        scope = {'__name__': 'editorial_smoke'}
        exec(compile(code, str(TOPO), 'exec'), scope)
        logits = torch.randn(1, 1, 16, 16, requires_grad=True)
        target = torch.zeros(1, 1, 16, 16); target[:, :, 7:10, 2:14] = 1
        loss = scope['SoftClDiceLoss']()(logits.sigmoid(), target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

if __name__ == '__main__':
    unittest.main()
