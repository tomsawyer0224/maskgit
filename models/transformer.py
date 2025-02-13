import sys

from einops import rearrange

if not "." in sys.path:
    sys.path.append(".")
import math

# import random
import torch
import torch.nn as nn

# from einops import rearrange


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_layers: int = 12,
        codebook_size: int = 1024,
        n_classes: int = 1000,
        latent_image_size: int = 16,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.latent_image_size = latent_image_size

        vocab_size = codebook_size + 1 + n_classes + 1
        self.CB_MASK_TOKEN_ID = codebook_size
        self.CLS_MASK_TOKEN_ID = codebook_size + 1 + n_classes
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        # assume: codebook_size = 1024, n_classes = 1000
        # num_embeddings = codebook_size + 1 + n_classes + 1 because the vocab contains:
        # 1024 tokens 'CB_id_0', ..., 'CB_id_1023' -> codebook tokens
        # 1 token 'CB_MASK' for masking codebook_id purpose
        # 1000 tokens 'CLS_0', ..., 'CLS_999' -> class tokens
        # 1 token 'CLS_MASK' for masking class_id purpose
        # vocab = ['CB_id_0', ..., 'CB_id_1023', 'CB_MASK', 'CLS_0', ..., 'CLS_999', 'CLS_MASK']
        # token_id = [0, 1, ..., 1023, 1024, 1024+1, ..., 1024+1000, 2025]

        self.positional_embedding = nn.init.trunc_normal_(
            tensor=nn.Parameter(torch.zeros(1, latent_image_size**2 + 1, d_model)),
            a=0.0,
            b=0.02,
        )
        self.TE_in = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.GELU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.TE = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.TE_out = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
        )

        self.bias = nn.Parameter(torch.zeros(latent_image_size**2 + 1, vocab_size))

    def forward(
        self,
        image_token: torch.LongTensor,
        image_token_mask: torch.BoolTensor = None,
        label: torch.LongTensor = None,
        label_mask: torch.BoolTensor = None,
    ):
        """
        args:
            image_token: (b, hw), h=w=latent_image_size, value = [0,...,hw-1]
            label: (b,), value = [0,...,n_classes-1]
            label_mask: (b,), value = True -> mask, value = False -> unmask
            all of these shoule be in the same device
        """
        b, hw = image_token.shape
        if image_token_mask is None:
            image_token_mask = torch.zeros(b, hw)

        if label is None:
            label_token = torch.tensor([self.CLS_MASK_TOKEN_ID] * b)
        else:
            label_token = label + self.codebook_size + 1
        if label_mask is None:
            label_mask = torch.tensor([False] * b)

        label_token = label_token.type_as(image_token)
        org_inp_seq = torch.cat(
            [image_token, label_token[:, None]], dim=-1
        )  # (b, hw + 1)

        # masks image_token
        image_token_mask = image_token_mask.type_as(image_token).bool()
        image_token[image_token_mask] = self.CB_MASK_TOKEN_ID

        # masks label_token
        label_mask = label_mask.type_as(label_token).bool()
        label_token[label_mask] = self.CLS_MASK_TOKEN_ID

        masked_inp_seq = torch.cat(
            [image_token, label_token[:, None]], dim=-1
        )  # (b, hw + 1)

        x = self.token_embedding(masked_inp_seq) + self.positional_embedding

        x = self.TE_in(x)
        x = self.TE(x)
        x = self.TE_out(x)

        logit = torch.matmul(x, self.token_embedding.weight.T) + self.bias
        # (b, hw + 1, vocab_size)
        return {
            "logit": logit,
            "org_inp_seq": org_inp_seq,
            "masked_inp_seq": masked_inp_seq,
            "image_token_mask": image_token_mask,
            "label_mask": label_mask,
        }

    def loss_fn(
        self,
        logit,
        org_inp_seq,
        masked_inp_seq,
        image_token_mask,
        label_mask,
        mask_only=False,
    ):
        """
        args:
            mask_only: if True -> calculates loss for masked tokens only
                       if False -> calculates loss for all tokens
        """
        if not mask_only:  # loss for all tokens
            tgt = org_inp_seq  # (b, hw + 1)
            inp = logit.permute(0, 2, 1)
            # (b, hw + 1, vocab_size) -> (b, vocab_size, hw + 1)
            loss = nn.functional.cross_entropy(inp, tgt, reduction="none").mean()
            # a trict (reduction = 'none' then .mean()) because cross_entropy doesn't work
            # with reproducibility (set in the traing phase)
        else:  # loss for masked tokens only
            b, s, v = logit.shape  # s = hw + 1
            logit_image = logit[:, :-1, :]  # (b, hw, vocab_size)
            logit_label = logit[:, -1, :]  # (b, vocab_size)

            masked_logit_image = logit_image[image_token_mask]

            masked_logit_image = masked_logit_image.reshape(b, -1, v)
            # (b, s1, vocab_size)
            masked_logit_image = masked_logit_image.permute(0, 2, 1)
            org_image_token = org_inp_seq[:, :-1]
            gt_image_token = org_image_token[image_token_mask]
            gt_image_token = gt_image_token.reshape(b, -1)

            # (b, s1)
            loss_img = nn.functional.cross_entropy(masked_logit_image, gt_image_token)

            masked_logit_label = logit_label[label_mask]  # (s2, vocab_size)
            org_label_token = org_inp_seq[:, -1]  # (s2,)
            gt_label_token = org_label_token[label_mask]  # (s2,)
            loss_label = nn.functional.cross_entropy(
                masked_logit_label, gt_label_token, reduction="none"
            ).mean()

            loss = 0.5 * (loss_img + loss_label)

        return loss

    @torch.no_grad
    def sequence_to_logit(
        self,
        image_token: torch.Tensor,
        label: torch.Tensor | None = None,
        return_logit_image: bool = True,
    ) -> torch.Tensor:
        """
        args:
            image_token: (b, hw)
            label: (b,)
            return_logit_image: if True, return logit of image tokens
                                if False, return all logits
        returns:
            logit: (b, hw, vocab_size) or (b, hw + 1, vocab_size)
        """
        self.eval()
        b, hw = image_token.shape
        if label is None:
            label = torch.tensor([0] * b).type_as(image_token)
        output = self(
            image_token=image_token, image_token_mask=None, label=label, label_mask=None
        )
        logit = output["logit"]
        self.train()
        if return_logit_image:
            return logit[:, :-1, :]
        return logit

    def masking_ratio_schedule(self, r: float, mode="cosine") -> float:
        # assert mode in ['uniform', 'cosine', 'square', 'cubic'], \
        # f"mode should be in ['uniform', 'cosine', 'square', 'cubic']"

        if mode == "uniform":
            mask_ratio = 1.0 - r
        elif mode == "cosine":
            mask_ratio = math.cos(r * math.pi / 2.0)
        elif mode == "square":
            mask_ratio = 1 - r**2
        elif mode == "cubic":
            mask_ratio = 1 - r**3
        else:
            raise NotImplementedError
        if mask_ratio < 1e-8:
            mask_ratio = 0.0
        return mask_ratio

    def get_mask(self, sequence: torch.Tensor, ratio: float) -> torch.BoolTensor:
        """
        gets random mask for sequence
        args:
            sequence: (b, seq_len) or (b,) -> original sequence
            ratio: range (0,1]
        returns:
            mask
        """
        # get number of indices to mask, (minimum = 1)
        k = int(ratio * sequence.shape[-1])
        mask = torch.zeros_like(sequence)
        if ratio == 0:
            return mask.bool()
        else:  # ratio > 0
            k = max(1, k)
        n_dim = mask.ndim
        if n_dim == 1:
            mask = mask[None, :]

        b, s = mask.shape
        if k > s:
            k = s
        indices_to_mask = []
        for i in range(b):
            # idx = torch.randint(low=0,high=s,size=(k,))
            # while len(set(idx.numpy())) != k:
            # idx = torch.randint(low=0,high=s,size=(k,))
            idx = torch.randperm(s)[0:k]
            indices_to_mask.append(idx)
            # mask[i][idx] = 1.0
        indices_to_mask = torch.stack(indices_to_mask).type_as(mask)
        mask.scatter_(dim=1, index=indices_to_mask, src=torch.ones_like(mask))

        mask = mask.bool()
        if n_dim == 1:
            mask = mask.squeeze()
        return mask

    def rand_multinomial(
        self,
        logit: torch.Tensor,
        clamp_value: tuple[int, int] | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        args:
            logit: (b, seq_len, vocab_size)
            clamp_value: [min, max] or None
            temperature: lower mean smoother (probability)
        returns:
            torch.LongTensor of shape (b, seq_len), value = [0, vocab_size)
        """
        prob = nn.functional.softmax(temperature * logit, dim=-1)
        b, s, v = prob.shape
        prob = rearrange(prob, "b s v -> (b s) v")
        sample = torch.multinomial(prob, 1).squeeze()
        sample = rearrange(sample, "(b s) -> b s", b=b)
        if clamp_value is not None:
            sample = sample.clamp(clamp_value[0], clamp_value[1])
        return sample

    def get_mask_of_low_confidence_token(
        self,
        mask_len: int | torch.Tensor,
        token_prob: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        args:
            mask_len: int or tensor of shape (b,)
            token_prob: (b, seq_len) value = probability of each token
                        token probability is not token confidence
        returns:
            bool tensor of shape (b, seq_len)
        """
        b, s = token_prob.shape
        if isinstance(mask_len, int):
            mask_len = torch.tensor([mask_len] * b).type_as(token_prob).long()
        mask_len = mask_len[:, None]
        g = torch.distributions.gumbel.Gumbel(0, 1)
        confidence = torch.log(token_prob) + temperature * g.sample(
            token_prob.shape
        ).type_as(token_prob)
        sorted_confidence = torch.sort(confidence, dim=-1)[0]
        cut_off = torch.gather(sorted_confidence, -1, mask_len)
        mask = confidence < cut_off
        # print(f'in the function get_mask_of_low_confidence_token')
        # print(f'confidence: \n{confidence}')
        # print(f'cut_off: \n{cut_off}')
        return mask

    def get_prob_of_token(
        self, logit: torch.Tensor, token_id: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:  # error here
        """
        args:
            logit: (b, seq_len, vocab_size)
            token_id: (b, seq_len)
        returns:
            probability of tokens
        """
        prob = nn.functional.softmax(temperature * logit, dim=-1)
        token_id = token_id[..., None]
        token_prob = torch.gather(prob, dim=-1, index=token_id).squeeze(-1)
        return token_prob

    @torch.no_grad
    def unmask(
        self,
        masked_image_token: torch.LongTensor,
        label: torch.LongTensor | None = None,
        n_step: int = 12,
        temperature: float = 1.0,
        masking_method: str = "cosine",
    ) -> torch.LongTensor:
        """
        args:
            unmasked_image_token: (b, hw), image token sequence which is masked some tokens
            label: (b,) label of images or None
        returns:
            LongTensor of shape (b, hw)
        """
        total_masked = torch.sum(masked_image_token == self.CB_MASK_TOKEN_ID, dim=-1)
        unmasked_seqs = []
        cur_seq = masked_image_token  # .clone()
        for step in range(n_step):
            # gets logit of image tokens (b, hw, vocab_size)
            logit = self.sequence_to_logit(
                image_token=cur_seq, label=label, return_logit_image=True
            )

            # gets random predicted image sequence
            random_pred_seq = self.rand_multinomial(
                logit=logit, clamp_value=[0, self.codebook_size - 1], temperature=1.0
            )

            # replaces masked tokens in cur_seq by random_pred_seq
            cur_mask = cur_seq == self.CB_MASK_TOKEN_ID
            pred_seq = torch.where(cur_mask, random_pred_seq, cur_seq)
            unmasked_seqs.append(pred_seq)

            # gets probability of tokens in pred_seq
            pred_seq_prob = self.get_prob_of_token(
                logit=logit, token_id=pred_seq, temperature=1.0
            )
            # set probability of unmasked tokens to infinity -> don't mask it again
            pred_seq_prob = torch.where(
                cur_mask, pred_seq_prob, torch.zeros_like(pred_seq_prob) + torch.inf
            )

            # gets mask_len for the next step, keep at least 1 unmasked token
            # masks at least 1 token for the next step
            ratio = 1.0 * (step + 1) / n_step
            mask_ratio = self.masking_ratio_schedule(r=ratio, mode=masking_method)
            mask_len = torch.floor(mask_ratio * total_masked).long()  # (b,)
            # print(f'in the unmask method')
            # print(f'step: {step}, mask_len before clamp: {mask_len}')
            n_cur_masked = torch.sum(
                cur_mask, dim=-1
            )  # number of current masked tokens
            min_ = torch.ones_like(mask_len)
            max_ = torch.where(n_cur_masked <= 1, 1, n_cur_masked - 1)
            mask_len = torch.clamp(
                mask_len,
                min_,
                max_,
                # torch.ones_like(mask_len), # masks at least 1 token
                # torch.sum(cur_mask, dim = -1) - 1 # keeps at least 1 unmasked token -> error here
            )
            # print(f'number of current masked tokens: {n_cur_masked}')
            # print(f'step: {step}, mask_len: {mask_len}')

            # masks low confidence tokens for the next round
            low_confidence_mask = self.get_mask_of_low_confidence_token(
                mask_len=mask_len,
                token_prob=pred_seq_prob,
                temperature=temperature * (1.0 - ratio),
            )
            cur_seq = torch.where(low_confidence_mask, self.CB_MASK_TOKEN_ID, pred_seq)
        unmasked_seqs = torch.stack(unmasked_seqs)
        return unmasked_seqs

    @torch.no_grad
    def generate_image_token(
        self,
        n_samples: int = 4,
        label: torch.Tensor | None = None,
        n_step: int = 12,
        temperature: float = 1.0,
        masking_method: str = "cosine",
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.to(device)
        masked_image_token = (
            torch.ones(n_samples, self.latent_image_size**2) * self.CB_MASK_TOKEN_ID
        )
        masked_image_token = masked_image_token.long().to(device)
        generated_sequence = self.unmask(
            masked_image_token=masked_image_token,
            label=label,
            n_step=n_step,
            temperature=temperature,
            masking_method=masking_method,
        )[-1]
        self.train()
        return generated_sequence
