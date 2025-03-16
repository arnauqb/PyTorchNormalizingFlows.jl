export make_real_nvp_flow_torch, make_masked_affine_autoregressive_flow_torch, make_neural_spline_flow_torch, make_planar_flow_torch
export serialize_flow, deserialize_flow

function make_real_nvp_flow_torch(dim, n_layers, hidden_dim)
    flows = []
    for i in 1:n_layers
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = normflows.nets.MLP(
            [Int(dim / 2), hidden_dim, hidden_dim, dim], init_zeros=true)
        # Add flow layer
        push!(flows, normflows.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        push!(flows, normflows.flows.Permute(dim, mode="swap"))
    end
    base = normflows.distributions.base.DiagGaussian(dim)
    # Construct flow model
    flow_py = normflows.NormalizingFlow(base, flows)
    for param in collect(flow_py.parameters())
        param.requires_grad = false
    end
    return PyTorchFlow(flow_py)
end

function _make_masked_affine_autoregressive_flow_torch(; dim, n_layers, n_units)
    flows = []
    for i in 1:n_layers
        push!(flows,
            normflows.flows.MaskedAffineAutoregressive(dim, n_units, num_blocks=2))
        push!(flows, normflows.flows.LULinearPermute(dim))
    end
    q0 = normflows.distributions.DiagGaussian(dim)
    nfm = normflows.NormalizingFlow(q0=q0, flows=flows)
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end
    return nfm
end

function make_masked_affine_autoregressive_flow_torch(; dim, n_layers, n_units)
    nfm = _make_masked_affine_autoregressive_flow_torch(; dim, n_layers, n_units)
    # remove the gradient from the parameters
    flow = PyTorchFlow(nfm)
    return flow
end

function make_neural_spline_flow_torch(dim, n_layers, hidden_units, hidden_layers=2)
    # Define flows
    K = n_layers

    latent_size = dim
    flows = []
    for i in 1:K
        push!(flows,
            normflows.flows.AutoregressiveRationalQuadraticSpline(
                latent_size, hidden_layers, hidden_units))
        push!(flows, normflows.flows.LULinearPermute(latent_size))
    end
    # Set base distribuiton
    q0 = normflows.distributions.DiagGaussian(dim, trainable=false)
    # Construct flow model
    nfm = normflows.NormalizingFlow(q0=q0, flows=flows)
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end

    return PyTorchFlow(nfm)
end

function make_planar_flow_torch(dim, n_layers)
    flows = []
    for i in 1:n_layers
        push!(flows, normflows.flows.Planar((dim,), act="leaky_relu"))
    end
    q0 = normflows.distributions.DiagGaussian(dim, trainable=false)
    nfm = normflows.NormalizingFlow(q0=q0, flows=flows)
    for param in collect(nfm.parameters())
        param.requires_grad = false
    end
    return PyTorchFlow(nfm)
end

## Serialization


function serialize_flow(
    flow, hyper_parameters, function_call)
    transform = flow.transform
    flow_parameters = flow.dist.params_flat
    buffers_arrays = [b.numpy() for b in flow.dist.buffers]
    params_arrays = [p.numpy() for p in flow.dist.params]
    dict = Dict(:transform => transform, :params_flat => flow.dist.params_flat,
        :hyper_parameters => hyper_parameters, :function_call => function_call,
        :buffers => buffers_arrays, :params => params_arrays, :indices => flow.dist.indices)
    return dict
end

function save_flow(serialized_dict, path)
    serialize(path, serialized_dict)
end

function deserialize_flow(flow_serialized::Dict)
    transform = flow_serialized[:transform]
    params = flow_serialized[:params]
    params = tuple([torch.tensor(p, dtype=torch.float) for p in params]...)
    params_flat = flow_serialized[:params_flat]
    hyper_parameters = flow_serialized[:hyper_parameters]
    function_call = flow_serialized[:function_call]
    buffers = flow_serialized[:buffers]
    buffers = tuple([torch.tensor(b, dtype=torch.float) for b in buffers]...)
    indices = flow_serialized[:indices]
    flow = _make_masked_affine_autoregressive_flow_torch(; hyper_parameters...)
    func_sampler, func_logpdf, _, _ = py"deconstruct_flow"(flow)
    full_flow = PyTorchFlow(
        func_sampler, func_logpdf, buffers, indices, params, params_flat, py"int"(flow.q0.d))
    return transformed(full_flow, transform)
end

function deserialize_flow(path::String)
    flow_serialized = deserialize(path)
    return deserialize_path(flow_serialized)
end