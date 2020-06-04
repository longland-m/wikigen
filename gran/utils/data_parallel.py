import operator
import torch
import warnings
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply


def _check_balance(device_ids):
  imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""

  dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

  def warn_imbalance(get_prop):
    values = [get_prop(props) for props in dev_props]
    min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
    max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
    if min_val / max_val < 0.75:
      warnings.warn(
          imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
      return True
    return False

  if warn_imbalance(lambda props: props.total_memory):
    return
  if warn_imbalance(lambda props: props.multi_processor_count):
    return


class DataParallel(Module):
  r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

  def __init__(self,
               module,
               device_ids=None,
               output_device=None,
               dim=0,
               gather_output=True):
    super(DataParallel, self).__init__()

    if not torch.cuda.is_available():
      self.module = module
      self.device_ids = []
      return

    if device_ids is None:
      device_ids = list(range(torch.cuda.device_count()))
    if output_device is None:
      output_device = device_ids[0]
    self.dim = dim
    self.module = module
    self.device_ids = device_ids
    self.output_device = output_device

    _check_balance(self.device_ids)

    if len(self.device_ids) == 1:
      self.module.cuda(device_ids[0])
    self.gather_output = gather_output

  def forward(self, *inputs, **kwargs):
    if not self.device_ids:
      return self.module(*inputs, **kwargs)
    assert kwargs == {}, 'not implemented'
    kwargs = [{} for _ in range(len(inputs))]
    if len(self.device_ids) == 1:
      return self.module(*inputs[0], **kwargs[0])
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, kwargs)
    if self.gather_output or len(self.device_ids) == 1:
      return self.gather(outputs, self.output_device)
    else:
      return outputs

  def replicate(self, module, device_ids):
    return replicate(module, device_ids)

  def scatter(self, inputs, kwargs, device_ids):
    return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

  def parallel_apply(self, replicas, inputs, kwargs):
    return parallel_apply(replicas, inputs, kwargs,
                          self.device_ids[:len(replicas)])

  def gather(self, outputs, output_device):
    return gather(outputs, output_device, dim=self.dim)
