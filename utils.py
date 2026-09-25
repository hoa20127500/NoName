import logging
import torch


def set_logger(log_file):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def scatter_mean(src, index, dim=-1):
    """Mean-reduce ``src`` along ``dim`` by ``index``. Replaces torch_scatter."""
    if index.numel() == 0:
        return src
    dim_size = int(index.max().item()) + 1
    out_size = list(src.shape)
    out_size[dim] = dim_size
    out = src.new_zeros(out_size)
    return out.scatter_reduce(dim, index, src, reduce='mean', include_self=False)


def scatter_sum(src, index, dim_size):
    if src.numel() == 0:
        return src.new_zeros((dim_size,) + src.shape[1:])
    out = src.new_zeros((dim_size,) + src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add(0, idx, src)


def scatter_softmax(src, index, dim_size):
    if src.numel() == 0:
        return src
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    maxes = src.new_full((dim_size,) + src.shape[1:], float('-inf'))
    maxes = maxes.scatter_reduce(0, idx, src, reduce='amax', include_self=True)
    maxes = torch.where(torch.isfinite(maxes), maxes, torch.zeros_like(maxes))
    src = src - maxes[index]
    exp = src.exp()
    den = scatter_sum(exp, index, dim_size).clamp_min(1e-9)
    return exp / den[index]