module PyTorchNormalizingFlows

using Bijectors
using ChainRulesCore
using Distributions
using DynamicPPL
using Functors
using PyCall
using Random

const torch = PyNULL()
const normflows = PyNULL()

function __init__()
    try
        copy!(torch, pyimport("torch"))
        copy!(normflows, pyimport("normflows"))
        @pyinclude(String(@__DIR__) * "/wrapper.py")
    catch e
        @warn "Failed to import PyTorch or normflows"
        @warn "Please check your Python environment"
    end
end

# Write your package code here.
include("dist.jl")
include("rules.jl")
include("flows.jl")
include("bijectors.jl")
end
