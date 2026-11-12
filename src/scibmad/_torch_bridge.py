import numpy as np
import torch
from ._runtime import jl 

from ._torch_autodiff import AUTODIFF_JULIA_CODE

# Initialize Julia differentiation helpers only
# when the PyTorch bridge is imported.
jl.seval(AUTODIFF_JULIA_CODE)

def _resolve_backend(name):
    """Resolve the requested Julia AD backend."""
    if not isinstance(name, str):
        raise TypeError("Backend must be a string")

    name = name.lower()

    if name == "forwarddiff":
        return jl.seval(
            "DifferentiationInterface.AutoForwardDiff()"
        )

    if name == "gtpsa":
        return jl.seval(
            "DifferentiationInterface.AutoGTPSA()"
        )

    raise ValueError(
        f"Unsupported AD backend: {name!r}. "
        "Supported backends: forwarddiff, gtpsa"
    )

def _to_julia_input(x):
    """
    Converts a torch tensor into a plain value that Julia/juliacall can consume, for use as an argument to a Julia function call

    Arguments:
    - x: the tensor to convert (tensor is detached before conversion, so Julia does not participate in PyTorch’s forward computation graph)

    Returns:
    - value: the converted value — a plain Python float if x has a single element, otherwise a flattened 1-D NumPy array
    - is_scalar_input: True if x had a single element, False otherwise
    """
    x_np = x.detach().cpu().numpy()

    # Only genuine 0-dimensional tensors become scalars.
    if x_np.ndim == 0:
        return float(x_np.item()), True

    # Preserve vector semantics, including shape (1,).
    return x_np.reshape(-1), False


def _to_numpy_output(result):
    """
    Converts a raw value returned from a Julia function call into a NumPy array, tagging whether it was scalar or vector-valued.

    Arguments:
    - result: the raw value returned by a Julia function call (typically a Julia scalar or vector, auto-converted by juliacall)

    Returns:
    - value: the result as a plain Python float (scalar case) or a NumPy array (vector case)
    - is_vector_output: True if result was vector-valued, False if scalar
    """
    result_np = np.array(result)
    if result_np.ndim == 0:
        return float(result_np), False
    return result_np, True


class _JuliaFunction(torch.autograd.Function): # Custom torch.autograd.Function 

    @staticmethod
    def forward(ctx, julia_func, tensor_arg_indices, backend, gtpsa_tracking, *args):
        """Runs julia_func on args, converting any tensor arguments to Julia-friendly values first, and stashes everything backward will need to compute gradients later

        Arguments:
        - ctx: the autograd context object PyTorch provides, used to stash state for backward
        - julia_func: the underlying Julia function (or a Python closure binding one to fixed keyword arguments) to call
        - tensor_arg_indices: positions within args that are torch.Tensor instances and need conversion/differentiation support; all other positions are passed through to Julia unchanged
        - *args: the full positional argument list for julia_func, tensor and non-tensor arguments mixed in call order

        Returns:
        - result: the Julia function's output, converted to a torch.Tensor matching the dtype/device of the first tensor argument
        """
        tensor_arg_indices = tuple(tensor_arg_indices)
        tensor_args = tuple(args[index] for index in tensor_arg_indices)

        jl_args = list(args)
        tensor_metadata = {}

        for index, x in zip(tensor_arg_indices, tensor_args):
            jl_x, is_scalar_input = _to_julia_input(x)
            jl_args[index] = jl_x
            tensor_metadata[index] = {
                "jl_x": jl_x,
                "is_scalar_input": is_scalar_input,
            }

        result = julia_func(*jl_args)
        result_np, is_vector_output = _to_numpy_output(result)

        ctx.is_vector_output = is_vector_output
        ctx.backend = backend
        ctx.gtpsa_tracking = gtpsa_tracking
        ctx.julia_func = julia_func
        ctx.jl_args = tuple(jl_args)
        ctx.tensor_arg_indices = tensor_arg_indices
        ctx.tensor_metadata = tensor_metadata
        ctx.num_args = len(args)
        ctx.save_for_backward(*tensor_args)

        """
        ctx.backend = (
        jl.scibmad_get_tracking_backend()
        if tracking
        else jl.scibmad_get_backend()
        ) """

        return torch.tensor(
            result_np,
            dtype=tensor_args[0].dtype,
            device=tensor_args[0].device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient with respect to each tensor argument by calling one of four Julia differentiation helpers then contracts the result against grad_output via the chain rule
        Runs one full Julia round-trip per differentiable argument

        Arguments:
        - ctx: the context populated by forward, containing the saved tensors, the Julia function, its full argument list, which positions were tensors, and per-tensor metadata (converted Julia value + scalar/vector flag)
        - grad_output: the gradient of the final loss with respect to this node's output, supplied by PyTorch

        Returns:
        - grads: a tuple (None, None, *arg_grads), None for the two non-tensor leading arguments to forward, followed by one entry per original argument: a gradient tensor for each tensor argument, None for every non-tensor argument
        """
        tensor_args = ctx.saved_tensors
        backend = ctx.backend
        grad_output_np = grad_output.detach().cpu().numpy()
        grad_output_flat = grad_output_np.flatten()

        def grad_for_tensor_arg(tensor_arg_index, x):
            """
            Computes the gradient for a single tensor argument of the call, by looking up its pre-converted Julia value and scalar/vector flag, 
            calling the matching Julia differentiation helper (scibmad_derivative_arg / scibmad_gradient_arg / scibmad_jacobian_arg / scibmad_jacobian_scalar_arg),
            and contracting against grad_output when the output was vector-valued

            Arguments:
            - tensor_arg_index: the position of this argument within the original call to julia_func
            - x: the original (saved) tensor argument, used only for its dtype/device/shape on the way out

            Returns:
            - grad: a torch.Tensor gradient, reshaped to match x's shape
            """
            metadata = ctx.tensor_metadata[tensor_arg_index]
            jl_x = metadata["jl_x"]
            is_scalar_input = metadata["is_scalar_input"]

            if is_scalar_input:
                if ctx.is_vector_output:
                    jac_np = np.array(
                        jl.scibmad_jacobian_scalar_arg(
                            ctx.julia_func,
                            backend,
                            ctx.jl_args,
                            tensor_arg_index + 1,
                            jl_x,
                        )
                    ).reshape(-1)
                    grad = np.array([jac_np @ grad_output_flat])
                else:
                    grad = np.array([
                        float(
                            jl.scibmad_derivative_arg(
                                ctx.julia_func,
                                backend,
                                ctx.jl_args,
                                tensor_arg_index + 1,
                                jl_x,
                            )
                        )
                    ])
            else:
                if ctx.is_vector_output:
                    jac_np = np.array(
                        jl.scibmad_jacobian_arg(
                            ctx.julia_func,
                            backend,
                            ctx.jl_args,
                            tensor_arg_index + 1,
                            jl_x,
                        )
                    )
                    grad = jac_np.T @ grad_output_flat
                else:
                    grad = np.array(
                        jl.scibmad_gradient_arg(
                            ctx.julia_func,
                            backend,
                            ctx.jl_args,
                            tensor_arg_index + 1,
                            jl_x,
                        )
                    )

            if not ctx.is_vector_output:
                grad = grad * grad_output_np

            return torch.tensor(grad, dtype=x.dtype, device=x.device).reshape(x.shape)

        
        arg_grads = [None] * ctx.num_args

        if ctx.gtpsa_tracking and ctx.needs_input_grad[4 + 4]:
            from ._runtime import glue

            # Argument index 4 is the vector of magnet strengths.
            strengths = next(
                x for index, x in zip(ctx.tensor_arg_indices, tensor_args)
                if index == 4
            )

            # Extract first-order GTPSA coefficients directly
            # from SciBmad tracking.
            _, jac = glue.torch_track_gtpsa_jacobian(*ctx.jl_args)

            jac_np = np.asarray(jac, dtype=np.float64)

            expected_shape = (grad_output_flat.size, strengths.numel())
            if jac_np.shape != expected_shape:
                raise RuntimeError(
                    f"GTPSA Jacobian shape {jac_np.shape}, "
                    f"expected {expected_shape}"
                )

            grad = jac_np.T @ grad_output_flat

            arg_grads[4] = torch.as_tensor(
                grad,
                dtype=strengths.dtype,
                device=strengths.device,
            ).reshape(strengths.shape)

        for tensor_arg_index, x in zip(ctx.tensor_arg_indices, tensor_args):
            if not ctx.needs_input_grad[4 + tensor_arg_index]:
                continue

            # Magnet-strength gradients were already obtained
            # from direct GTPSA coefficient extraction.
            if ctx.gtpsa_tracking and tensor_arg_index == 4:
                continue

            arg_grads[tensor_arg_index] = grad_for_tensor_arg(
                tensor_arg_index, x
            )

        return None, None, None, None, *arg_grads

class _JuliaTorchCallable: # wraps Julia-callable so tensor arguments get autograd support, plain arguments go straight to Julia w/ no autograd overhead
    
    __slots__ = ("_fn", "_tracking", "_backend", "_gtpsa_tracking")

    def __init__(self, fn, tracking=False, backend="forwarddiff"):
        """
        Stores the wrapped Julia callable

        Arguments:
        - fn: a Julia function/value (typically a juliacall AnyValue) pulled from the live Julia session

        Returns:
        - (none)
        """
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_tracking", tracking)
        object.__setattr__(self, "_backend", _resolve_backend(backend))

        object.__setattr__(
            self,
            "_gtpsa_tracking",
            tracking and backend.lower() == "gtpsa",
        )

    def __call__(self, *args, **kwargs):
        """
        Call the wrapped Julia function, routes through _JuliaFunction.apply for autograd if any positional arg is a Tensor, else calls Julia directly
        Tensor kwargs aren't supported, apply() only accepts positional args

        Arguments:
        - *args: positional arguments to forward to the Julia function; any torch.Tensor among these triggers the autograd-aware path
        - **kwargs: keyword arguments to forward; raises if any value is a torch.Tensor while a positional tensor argument is also present

        Returns:
        - result: either the raw Julia/NumPy result (no-tensor path) or a differentiable torch.Tensor (tensor path)
        """
        fn = object.__getattribute__(self, "_fn")
        tracking = object.__getattribute__(self, "_tracking")
        backend = object.__getattribute__(self, "_backend")
        gtpsa_tracking = object.__getattribute__(self, "_gtpsa_tracking")

        tensor_arg_indices = [
            index for index, arg in enumerate(args)
            if isinstance(arg, torch.Tensor)
        ]

        if tensor_arg_indices:
            tensor_kwargs = [
                name for name, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            ]
            if tensor_kwargs:
                names = ", ".join(sorted(tensor_kwargs))
                raise TypeError(
                        f"Tensor keyword arguments are not supported yet: {names}"
                    )
            
            if kwargs:
                bound_fn = jl.scibmad_bind_kwargs(fn, kwargs)

                return _JuliaFunction.apply(
                    bound_fn,
                    tuple(tensor_arg_indices),
                    backend,
                    gtpsa_tracking,
                    *args
                )

            return _JuliaFunction.apply(
                fn,
                tuple(tensor_arg_indices),
                backend,
                gtpsa_tracking,
                *args
            )
        return fn(*args, **kwargs)

    def __getattr__(self, name):
        """
        Forwards attribute lookups to the wrapped Julia function, so a wrapped callable still exposes whatever attributes the underlying Julia value has

        Arguments:
        - name: the attribute name being looked up

        Returns:
        - value: the corresponding attribute from the wrapped Julia function
        """
        return getattr(object.__getattribute__(self, "_fn"), name)

    def __setattr__(self, name, value):
        """
        Forwards attribute assignment to the wrapped Julia function.

        Arguments:
        - name: the attribute name being set
        - value: the value to assign

        Returns:
        - (none)
        """
        setattr(object.__getattribute__(self, "_fn"), name, value)

    def __repr__(self):
        """
        Returns the wrapped Julia function's own repr, rather than a generic wrapper repr, so the wrapping stays invisible when inspecting objects

        Arguments:
        - (none)

        Returns:
        - repr_str: the repr of the wrapped Julia function
        """
        return repr(object.__getattribute__(self, "_fn"))


#### """DA STUFF"""

class _JuliaValueGradFunction(torch.autograd.Function):
    """
    Julia function must return:
        value, gradient

    where gradient is d(value)/d(x).
    """

    @staticmethod
    def forward(ctx, julia_func, x):
        jl_x, _ = _to_julia_input(x)

        value, grad = julia_func(jl_x)

        grad_np = np.asarray(grad, dtype=float)

        ctx.save_for_backward(
            torch.tensor(
                grad_np,
                dtype=x.dtype,
                device=x.device,
            ).reshape(x.shape)
        )

        return torch.tensor(
            float(value),
            dtype=x.dtype,
            device=x.device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (saved_grad,) = ctx.saved_tensors

        return None, grad_output * saved_grad

class _JuliaValueGradCallable:
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return _JuliaValueGradFunction.apply(self._fn, x)

        return self._fn(x)