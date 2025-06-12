import torch
from typing import Tuple, List
from torch._vmap_internals import _vmap

from torch.autograd.functional import _as_tuple, _autograd_grad, _grad_preprocess, _check_requires_grad, _validate_v, \
    _fill_in_zeros, _grad_postprocess, _tuple_postprocess


############### rewrite functions of 'vjp' and 'jvp', and replace the 'func' with 'output', the output of func. ###############

def vjp(outputs, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between a vector ``v`` and the
    Jacobian of the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector
            Jacobian product is computed.  Must be the same size as the output
            of ``func``. This argument is optional when the output of ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vjp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4)
        >>> vjp(exp_reducer, inputs, v)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782]),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]]))

        >>> vjp(exp_reducer, inputs, v, create_graph=True)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782], grad_fn=<SumBackward1>),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]], grad_fn=<MulBackward0>))

        >>> def adder(x, y):
        ...   return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = torch.ones(2)
        >>> vjp(adder, inputs, v)
        (tensor([2.4225, 2.3340]),
         (tensor([2., 2.]), tensor([3., 3.])))
    """

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        # outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "vjp")
        _check_requires_grad(outputs, "outputs", strict=strict)

        if v is not None:
            _, v = _as_tuple(v, "v", "vjp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, outputs, is_outputs_tuple)
        else:
            if len(outputs) != 1 or outputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the "
                                   "user-provided function returns "
                                   "a single Tensor with a single element.")

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        vjp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    vjp = _grad_postprocess(vjp, create_graph)

    return _tuple_postprocess(vjp, is_inputs_tuple)


def jvp(outputs, inputs, v=None, create_graph=False, strict=False):
    r"""Function that computes the dot product between  the Jacobian of
    the given function at the point given by the inputs and a vector ``v``.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Jacobian
            vector product is computed. Must be the same size as the input of
            ``func``. This argument is optional when the input to ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            jvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the output.

    Example:

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4, 4)
        >>> jvp(exp_reducer, inputs, v)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106]),
         tensor([6.3090, 4.6742, 7.9114, 8.2106]))

        >>> jvp(exp_reducer, inputs, v, create_graph=True)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SumBackward1>),
         tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SqueezeBackward1>))

        >>> def adder(x, y):
        ...   return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.ones(2), torch.ones(2))
        >>> jvp(adder, inputs, v)
        (tensor([2.2399, 2.5005]),
         tensor([5., 5.]))

    Note:
        The jvp is currently computed by using the backward of the backward
        (sometimes called the double backwards trick) as we don't have support
        for forward mode AD in PyTorch at the moment.
    """

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
        # inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "jvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError("The vector v can only be None if the input to "
                                   "the user-provided function is a single Tensor "
                                   "with a single element.")

        # outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jvp")
        _check_requires_grad(outputs, "outputs", strict=strict)
        # The backward is linear so the value of grad_outputs is not important as
        # it won't appear in the double backward graph. We only need to ensure that
        # it does not contain inf or nan.
        grad_outputs = tuple(torch.zeros_like(out, requires_grad=True) for out in outputs)

        grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
        _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    if create_graph:
        with torch.enable_grad():
            grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph)
            jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")
    else:
        grad_res = _autograd_grad(grad_inputs, grad_outputs, v, create_graph=create_graph)
        jvp = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_graph)
    jvp = _grad_postprocess(jvp, create_graph)

    return _tuple_postprocess(jvp, is_outputs_tuple)



if __name__ == 'main':
    def exp_reducer(x):
        return x.exp().sum(dim=1)
    inputs = torch.rand(4, 4)
    v = torch.ones(4, 4)
    res = torch.autograd.functional.jvp(exp_reducer, inputs, v, create_graph=True)
    print('Done.')