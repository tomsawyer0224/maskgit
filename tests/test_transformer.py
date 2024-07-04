import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest
import torch
import torch.nn as nn
from models import transformer

class Test_Transformer(unittest.TestCase):
    def setUp(self):
        self.latent_image_size = 8
        self.codebook_size = 128
        self.n_classes = 3
        self.trans = transformer.Transformer(
            d_model = 4,
            nhead = 1,
            dim_feedforward = 2048,
            dropout = 0.1,
            activation = 'gelu',
            num_layers = 6,
            codebook_size = self.codebook_size,
            n_classes = self.n_classes,
            latent_image_size = self.latent_image_size,
        )
    def test_forward(self):
        print('---test_forward---')
        image_token = torch.randint(
            0,self.codebook_size,(4,self.latent_image_size**2)
        )
        #image_token_mask = torch.randint(0,2,image_token.size()).bool()
        image_token_mask = torch.zeros(image_token.size()).bool()
        b, hw = image_token_mask.shape
        for i in range(b):
            idx = torch.randint(0,hw, (5,))
            while len(set(idx.numpy())) != 5:
                idx = torch.randint(0,hw, (5,))
            image_token_mask[i][idx] = True

        label = torch.randint(0,self.n_classes,(4,))
        label_mask = torch.randint(0,2,(4,)).bool()
        output = self.trans(
            image_token = image_token,
            image_token_mask = image_token_mask,
            label = label,
            label_mask = label_mask
        )
        for k, v in output.items():
            print(f'{k}: {v.shape}, {v.dtype}')
            print('+++'*10)
        print()
        print('---tets_loss_fn---')
        loss = self.trans.loss_fn(**output, mask_only = True)
        print(f'loss mask_only = True: {loss}, {loss.grad_fn}')
        print()
        loss = self.trans.loss_fn(**output, mask_only = False)
        print(f'loss mask_only = False: {loss}, {loss.grad_fn}')
        print()
    def test_masking_ratio_schedule(self):
        print('---test_masking_ratio_schedule---')
        ratios = [0,0.1,0.2,0.3,0.9,0.9999,1]
        r = [
            self.trans.masking_ratio_schedule(rat, mode = 'cosine') for rat in ratios
        ]
        print(f'ratios = {ratios}')
        print(f'r = {r}')
        print()
    def test_get_mask(self):
        print('---test_get_mask---')
        sequence_1d = torch.randint(20,50,(8,))
        sequence_2d = torch.randint(0,20,(8,10))
        mask_1d = self.trans.get_mask(sequence_1d, 0.1)
        mask_2d = self.trans.get_mask(sequence_2d, 0.1)
        print(f'mask_1d: \n{mask_1d}')
        print(f'mask_2d: \n{mask_2d}')
        print()
    def test_rand_multinomial(self):
        print('---test_rand_multinomial---')
        logit = torch.randn(4,6,10)
        temperature = 1.0
        clamp_value = None#[0,2]
        sample = self.trans.rand_multinomial(
            logit = logit,
            clamp_value = clamp_value,
            temperature = temperature
        )
        print(f'sample: \n{sample}')
        print()
    def test_get_mask_low_confidence_token(self):
        print('---test_get_mask_of_low_confidence_token---')
        logit = torch.randn(4,6)
        prob = nn.functional.softmax(logit,-1)
        mask_len = torch.tensor([2,3,2,1])
        mask = self.trans.get_mask_of_low_confidence_token(
            mask_len = mask_len,
            token_prob = prob,
            temperature = 1.0
        )
        print(f'prob: \n{prob}')
        print(f'mask: \n{mask}')
        print()
    def test_get_prob_of_token(self):
        print('---test_get_prob_of_token---')
        logit = torch.randn(4,6,10)
        token_id = torch.randint(0,10,(4,6))
        prob = self.trans.get_prob_of_token(
            logit=logit,
            token_id=token_id
        )
        print(f'logit: \n{logit}')
        print(f'token_id: \n{token_id}')
        print(f'prob: \n{prob}')
        print()
    def test_sequence_to_logit(self):
        print('---test_sequence_to_logit---')
        image_token = torch.randint(0,self.codebook_size,(4,self.latent_image_size**2))
        label = torch.randint(0,self.n_classes,(4,))
        logit = self.trans.sequence_to_logit(
            image_token = image_token,
            label = label,
            return_logit_image = True
        )
        print(f'logit: {logit.shape}')
    def test_unmask(self):
        print('---test_unmask---')
        #image_token = torch.randint(0,self.codebook_size,(4,self.latent_image_size**2))
        #mask = torch.ones_like(image_token)
        masked_image_token = torch.ones((4,self.latent_image_size**2))*self.trans.CB_MASK_TOKEN_ID
        masked_image_token = masked_image_token.long()
        label = torch.randint(0,self.n_classes,(4,))
        unmasked_seqs = self.trans.unmask(
            masked_image_token = masked_image_token,
            label = label,
            n_step = 12,
            temperature = 1.0,
            masking_method = 'cosine'
        )
        print(f'unmasked_seqs: {unmasked_seqs.shape}')
        print(f'predicted sequence: \n{unmasked_seqs[-1]}')
        print()
    def test_generate_image_token(self):
        print('---test_generate_image_token---')
        label = None
        generated_image_token = self.trans.generate_image_token(
            n_samples = 4,
            label = label
        )
        print(f'generated_image_token: \n{generated_image_token}')
        print()

if __name__=="__main__":
    unittest.main()