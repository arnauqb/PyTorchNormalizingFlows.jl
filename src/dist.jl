# defines the flow as a distribution

struct PyTorchFlow{T,N,M} <: Distributions.ContinuousMultivariateDistribution
    func_sampler::PyObject
    func_logpdf::PyObject
    buffers::NTuple{N,PyObject}
    indices::Vector{Tuple{Int64,Int64}}
    params::NTuple{M,PyObject}
    params_flat::Vector{T}
    output_length::Int64
end
Distributions.length(d::PyTorchFlow) = d.output_length
Functors.@functor PyTorchFlow (params_flat,)

function PyTorchFlow(flow::PyObject)
    func_sampler, func_logpdf, params, buffers = py"deconstruct_flow"(flow)
    params_flat, indices = py"flatten_params"(params)
    params_flat = params_flat.numpy()
    return PyTorchFlow(
        func_sampler, func_logpdf, buffers, indices, params, params_flat, py"int"(flow.q0.d))
end

function Distributions.rand(dist::PyTorchFlow)
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat), dist.indices, dist.params)
    return dist.func_sampler(params, dist.buffers, 1).flatten().numpy()
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow) = rand(dist)

function Distributions.rand(dist::PyTorchFlow, n::Int)
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat, dtype=torch.float), dist.indices, dist.params)
    return dist.func_sampler(params, dist.buffers, n).numpy()
end
Distributions.rand(rng::Random.AbstractRNG, dist::PyTorchFlow, n::Int) = rand(dist, n)

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractArray{<:Real})
    params = py"recover_flattened"(
        torch.tensor(dist.params_flat, dtype=torch.float), dist.indices, dist.params)
    return dist.func_logpdf(
        params, dist.buffers, torch.tensor(x, dtype=torch.float)).numpy()
end

function Distributions.logpdf(dist::PyTorchFlow, x::AbstractVector{<:Real})
    logpdf(dist, reshape(x, length(x), 1))[1]
end