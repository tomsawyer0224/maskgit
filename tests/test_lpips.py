import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest
import torch
import torch.nn as nn
#from models import lpips
from loss import lpips


class TestLPIPS(unittest.TestCase):
    def setUp(self):
        pass
    def test_LPIPS(self):
        print('---test_LPIPS---')
        lpips_loss_fn = lpips.LPIPS().eval()
        '''
        for ll in range(5):
            print(lpips_loss_fn.lins[ll].model[1].weight.flatten())
            print('+++'*20)
        return
        '''
        real_image = torch.rand(4,3,128,128, requires_grad = True)
        decoded_image = torch.randn(4,3,128,128, requires_grad = True)
        real_image = 2*real_image - 1
        #decoded_image = 2*decoded_image - 1
        perceptual_loss = lpips_loss_fn(real_image, decoded_image)
        print(perceptual_loss.mean())
        print(f'perceptual_loss: {perceptual_loss.shape}, {perceptual_loss.grad_fn}')
        print()
        rec_loss = torch.abs(real_image-decoded_image)
        print(f'rec_loss: {rec_loss.shape}')
        print()
        nll_loss = perceptual_loss + rec_loss
        print(nll_loss.mean())

if __name__=="__main__":
    unittest.main()
