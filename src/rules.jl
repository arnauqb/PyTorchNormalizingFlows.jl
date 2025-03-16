## Zygote rules
function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow, n::Int64)
    samples, vjp = py"make_vjp_sampler"(
        d.func_sampler, d.params, torch.tensor(d.params_flat, dtype=torch.float),
        d.buffers, d.indices, n)
    function rand_pullback(y_tangent)
        grad, = vjp(torch.tensor(y_tangent, dtype=torch.float))
        d_tangent = Tangent{PyTorchFlow}(; params_flat=grad.numpy())
        return NoTangent(), d_tangent, NoTangent()
    end
    return samples.numpy(), rand_pullback
end

function ChainRulesCore.rrule(::typeof(rand), d::PyTorchFlow)
    samples, vjp = py"make_vjp_sampler"(
        d.func_sampler, d.params, torch.tensor(d.params_flat, dtype=torch.float),
        d.buffers, d.indices, 1)
    function rand_pullback(y_tangent)
        grad, = vjp(torch.tensor(y_tangent, dtype=torch.float).reshape(-1, 1))
        d_tangent = Tangent{PyTorchFlow}(; params_flat=grad.numpy())
        return NoTangent(), d_tangent
    end
    return samples.numpy()[:], rand_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractVector{<:Real})
    lps, vjp = py"make_vjp_logpdf"(
        d.func_logpdf, d.params, torch.tensor(d.params_flat, dtype=torch.float), d.buffers,
        d.indices, torch.tensor(x, dtype=torch.float).reshape(-1, 1))
    function logpdf_pullback(y_tangent)
        grad_params, grad_x = vjp(torch.tensor(y_tangent, dtype=torch.float).reshape(1))
        d_tangent = Tangent{PyTorchFlow}(; params_flat=grad_params.numpy())
        x_tangent = grad_x.numpy()
        return NoTangent(), d_tangent, x_tangent
    end
    return lps.numpy()[1], logpdf_pullback
end

function ChainRulesCore.rrule(::typeof(logpdf), d::PyTorchFlow, x::AbstractMatrix{<:Real})
    lps, vjp = py"make_vjp_logpdf"(
        d.func_logpdf, d.params, torch.tensor(d.params_flat, dtype=torch.float),
        d.buffers, d.indices, torch.tensor(x))
    function logpdf_pullback(y_tangent)
        grad_params, grad_x = vjp(torch.tensor(y_tangent))
        d_tangent = Tangent{PyTorchFlow}(; params_flat=grad_params.numpy())
        x_tangent = grad_x.numpy()
        return NoTangent(), d_tangent, NoTangent(), x_tangent
    end
    return lps.numpy(), logpdf_pullback
end