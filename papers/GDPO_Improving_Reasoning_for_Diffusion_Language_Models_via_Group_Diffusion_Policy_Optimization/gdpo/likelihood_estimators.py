import abc
import torch
import torch.nn.functional as F

from tqdm import tqdm

class LikelihoodEstimator(abc.ABC):
    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id, attn_mask=None):
        if cfg_scale > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])
            attn_mask = torch.cat([attn_mask, attn_mask])

        logits = model(input_ids=batch, attention_mask=attn_mask).logits

        if cfg_scale > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    @abc.abstractmethod
    def get_log_likelihood(self, model, seq, attn_mask=None, logits_to_keep=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to compute log likelihood. Must be implemented by subclasses.
        Args:
            model: The model to use for likelihood estimation.
            prompt: The prompt tensor or input.
            answer: The answer tensor or input.
            **kwargs: Additional keyword arguments for the estimation method.
        Returns:
            A tuple of (likelihood, losses) or as defined by the subclass.
        """
        pass

class MCEstimator(LikelihoodEstimator):
    def __init__(self,verbose=False):
        self.verbose = verbose

    def forward_process(self, batch, prompt_index, mask_id):
        b, length = batch.shape

        target_len = (length - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        noisy_batch = torch.where(is_mask, mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, length)
    
    
    def get_log_likelihood(self, model, seq, attn_mask=None, logits_to_keep=None, num_batches=2, mc_batch_size=2, cfg_scale=0., mask_id=126336, seed=None):
        '''
        Args:
            model: Mask predictor.
            seq: A tensor of shape [B, L1]
            logits_to_keep: The elements that are relevant looked from right to left
            mc_num: Monte Carlo estimation times.
                    As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                    single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                    Monte Carlo samples are adequate to produce stable results.
            batch_size: Mini batch size.
            cfg_scale: Unsupervised classifier-free guidance scale.
            mask_id: The toke id of [MASK] is 126336.
        '''
        if seed is not None:
            torch.manual_seed(seed)
        bs, seq_len = seq.shape
        logits_to_keep = seq_len if logits_to_keep is None else logits_to_keep
        prompt_index = torch.arange(seq_len, device=model.device) < seq_len - logits_to_keep
        seq = seq.unsqueeze(1).repeat((1, mc_batch_size, 1)).reshape(-1, seq_len) # Shape [B * MC, L]
        big_attn = None
        if attn_mask is not None:
            big_attn = attn_mask.unsqueeze(1).expand(-1, mc_batch_size, -1).reshape(-1, seq_len) # Shape [B * MC, L]

        losses = torch.zeros((bs, num_batches), device=seq.device) 
        pbar = tqdm(range(num_batches), leave=False) if self.verbose else range(num_batches)
        for i in pbar:
            if self.verbose:
                pbar.set_description('Llada estimation')
            perturbed_seq, p_mask = self.forward_process(seq, prompt_index, mask_id)
            not_mask_index = (perturbed_seq != mask_id).reshape(bs, mc_batch_size, -1)

            logits = self.get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id, attn_mask=big_attn)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), seq.flatten(), reduction='none') / p_mask.flatten()
            loss = loss.reshape(bs, mc_batch_size, -1)
            loss[not_mask_index] = 0
            loss = loss.sum(dim=(1,2)) / mc_batch_size

            losses[:, i] = loss

        return - losses.mean(dim=-1), losses

class D1Estimator(LikelihoodEstimator):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def d1_forward_process(self, batch, prompt_index, mask_id, p_mask_prompt=0.15):
        b, length = batch.shape
        t_p = torch.ones(b, device=batch.device) * p_mask_prompt

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, length), device=batch.device)

        # For prompt tokens: mask if random_matrix < t_p
        # For completion tokens: always mask
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index  # all completion tokens are masked
        is_mask = is_mask_prompt | is_mask_completion

        # Create a noisy (masked) batch
        noisy_batch = torch.where(is_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),  # prompt token probability
            torch.ones_like(t_p).unsqueeze(1),  # completion token probability
        )

        return noisy_batch, p_mask



    def get_log_likelihood(self, model, seq, attn_mask=None, logits_to_keep=None,
            num_batches=1, mc_batch_size=1, cfg_scale=0., mask_id=126336, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        bs, seq_len = seq.shape
        logits_to_keep = seq_len if logits_to_keep is None else logits_to_keep
        prompt_index = torch.arange(seq_len, device=model.device) < seq_len - logits_to_keep
        seq = seq.unsqueeze(1).repeat((1, mc_batch_size, 1)).reshape(-1, seq_len) # Shape [B * MC, L]
        big_attn = None
        if attn_mask is not None:
            big_attn = attn_mask.unsqueeze(1).repeat((1, mc_batch_size, 1)).reshape(-1, seq_len) # Shape [B * MC, L]

        losses = torch.zeros((bs, num_batches), device=seq.device) 
        pbar = tqdm(range(num_batches), leave=False) if self.verbose else range(num_batches)
        for i in pbar:
            if self.verbose:
                pbar.set_description('D1 estimation')
            perturbed_seq, p_mask = self.d1_forward_process(seq, prompt_index, mask_id)
            not_mask_index = (perturbed_seq != mask_id).reshape(bs, mc_batch_size, -1)

            logits = self.get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id, attn_mask=big_attn)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), seq.flatten(), reduction='none') / p_mask.flatten()
            loss = loss.reshape(bs, mc_batch_size, -1)
            loss[not_mask_index] = 0
            loss = loss.sum(dim=(1,2)) / mc_batch_size

            losses[:, i] = loss

        return - losses.mean(dim=-1), losses

class NumericalIntegrationEstimator(LikelihoodEstimator):
    def __init__(self, mode='gauss-2', verbose=False):
        self.mode = mode
        self.verbose = verbose

        # Gauss-Legendre quadrature points and weights for [-1, 1] interval
        self.quadratures = {
            'gauss-1': {
                'points': [0.0],
                'weights': [2.0]
            },
            'gauss-2': {
                'points': [-(1/3**0.5), (1/3**0.5)],
                'weights': [1.0, 1.0]
            },
            'gauss-3': {
                'points': [-(3/5)**0.5, 0.0, (3/5)**0.5],
                'weights': [5/9, 8/9, 5/9]
            },
            'gauss-4': {
                'points': [
                    -((3/7 - 2/7 * (6/5)**0.5)**0.5),
                    +((3/7 - 2/7 * (6/5)**0.5)**0.5),
                    -((3/7 + 2/7 * (6/5)**0.5)**0.5),
                    +((3/7 + 2/7 * (6/5)**0.5)**0.5)
                ],
                'weights': [
                    (18 + 30**0.5)/36,
                    (18 + 30**0.5)/36,
                    (18 - 30**0.5)/36,
                    (18 - 30**0.5)/36
                ]
            },
            'gauss-5': {
                'points': [
                    0.0,
                    -(1/3) * (5 - 2*(10/7)**0.5)**0.5,
                    +(1/3) * (5 - 2*(10/7)**0.5)**0.5,
                    -(1/3) * (5 + 2*(10/7)**0.5)**0.5,
                    +(1/3) * (5 + 2*(10/7)**0.5)**0.5
                ],
                'weights': [
                    128/225,
                    (322 + 13*(70**0.5))/900,
                    (322 + 13*(70**0.5))/900,
                    (322 - 13*(70**0.5))/900,
                    (322 - 13*(70**0.5))/900
                ]
            }
        }
 
        self.points = self.quadratures[mode]['points']
        self.weights = self.quadratures[mode]['weights']

    def forward_process(self, batch, p_mask, prompt_index, mask_id):
        b, length = batch.shape

        target_len = (length - prompt_index.sum()).item()

        is_mask = torch.zeros((b, target_len), dtype=torch.bool, device=batch.device)
        for i in range(b):
            num_mask = int(p_mask * target_len)
            perm = torch.randperm(target_len).to(batch.device)[:num_mask]
            is_mask[i][perm] = 1

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        noisy_batch = torch.where(is_mask, mask_id, batch)

        return noisy_batch, p_mask
    
    
    def get_log_likelihood(self, model, seq, attn_mask=None, logits_to_keep=None, num_batches=1, mc_batch_size=1, cfg_scale=0., mask_id=126336, seed=None):
        '''
        Args:
            model: Mask predictor.
            seq: A tensor of shape [B, L1]
            logits_to_keep: The elements that are relevant looked from right to left
            mc_num: Monte Carlo estimation times.
                    As detailed in Appendix B.5. Since MMLU, CMMLU, and C-EVAL only require the likelihood of a single token, a
                    single Monte Carlo estimate is sufficient for these benchmarks. For all other benchmarks, we find that 128
                    Monte Carlo samples are adequate to produce stable results.
            batch_size: Mini batch size.
            cfg_scale: Unsupervised classifier-free guidance scale.
            mask_id: The toke id of [MASK] is 126336.
        '''
        if seed is not None:
            torch.manual_seed(seed)
        bs, seq_len = seq.shape
        logits_to_keep = seq_len if logits_to_keep is None else logits_to_keep
        prompt_index = torch.arange(seq_len, device=model.device) < seq_len - logits_to_keep
        seq = seq.unsqueeze(1).repeat((1, mc_batch_size, 1)).reshape(-1, seq_len) # Shape [B * MC, L]
        big_attn = None
        if attn_mask is not None: 
            big_attn = attn_mask.unsqueeze(1).expand(-1, mc_batch_size, -1).reshape(-1, seq_len) # Shape [B * MC, L]

        losses = torch.zeros((bs, num_batches), device=seq.device, ) 
        pbar = tqdm(range(num_batches), leave=False) if self.verbose else range(num_batches)
        for i in pbar:
            if self.verbose:
                pbar.set_description(f'{self.mode} Estimation')
            if self.mode == 'simpson':
                # T = .1
                loss_1, _ = self.get_loss(model, seq, bs, mc_batch_size, cfg_scale, mask_id, prompt_index, 0.1, attn_mask=big_attn)
                # T = .5
                loss_2, _ = self.get_loss(model, seq, bs, mc_batch_size, cfg_scale, mask_id, prompt_index, 0.5, attn_mask=big_attn)
                # T = 1.
                loss_3, _ = self.get_loss(model, seq, bs, mc_batch_size, cfg_scale, mask_id, prompt_index, 1.0, attn_mask=big_attn)

                # Standard Simpsons
                loss = (loss_1 + 4 * loss_2 + loss_3) / 6
            elif 'gauss-' in self.mode:
                loss = 0
                for point, weight in zip(self.points, self.weights):
                    xi = point * .5 + .5
                    loss_i, _ = self.get_loss(model, seq, bs, mc_batch_size, cfg_scale, mask_id, prompt_index, xi, attn_mask=big_attn)

                    loss = loss + weight * loss_i
                
                loss = .5 * loss # Change of variable from [-1, 1] to [0,1]

            losses[:, i] = loss

        return - losses.mean(dim=-1), losses

    def get_loss(self, model, seq, bs, mc_batch_size, cfg_scale, mask_id, prompt_index, p_mask, attn_mask=None):
        perturbed_seq, p_mask = self.forward_process(seq, p_mask, prompt_index, mask_id)
        not_mask_index = (perturbed_seq != mask_id).reshape(bs, mc_batch_size, -1)

        logits = self.get_logits(model, perturbed_seq, prompt_index, cfg_scale, mask_id, attn_mask=attn_mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), seq.flatten(), reduction='none') / p_mask # .flatten()
        loss = loss.reshape(bs, mc_batch_size, -1)
        loss[not_mask_index] = 0
        loss_per_mc = loss.sum(dim=2)
        loss = loss.sum(dim=(1,2)) / mc_batch_size

        return loss, loss_per_mc


def get_estimator(method) -> LikelihoodEstimator:
    if method == 'mc':
        estimator = MCEstimator()
    elif method == 'd1':
        estimator = D1Estimator()
    else:
        estimator = NumericalIntegrationEstimator(method)
    return estimator