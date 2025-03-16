##
using AdvancedVI
using ADTypes
using Bijectors
using DynamicPPL
using Distributions
using Optimisers
using PyTorchNormalizingFlows
using Test
using Random
using Zygote

Random.seed!(1234);

@model function test_model(data)
    mu ~ Uniform(0, 10)
    sigma ~ Uniform(0, 10)
    data ~ Normal(mu, sigma)
end

## setup true data
true_mu = 0.5;
true_sigma = 2.0;
data = rand(Normal(true_mu, true_sigma), 100);
model = test_model(data);

## setup flow transform to domain of the model with bijectors
bijector_transf = inverse(bijector(model));
flow = make_masked_affine_autoregressive_flow_torch(dim=2, n_layers=4, n_units=16);
flow_transformed = transformed(flow, bijector_transf);
flow_untrained = deepcopy(flow_transformed);

## run VI
n_montecarlo = 5;
lp = DynamicPPL.LogDensityFunction(model);
elbo = AdvancedVI.RepGradELBO(n_montecarlo, entropy=AdvancedVI.MonteCarloEntropy());
optimizer = Optimisers.AdamW(5e-4);
q, _, stats, _ = AdvancedVI.optimize(
    lp,
    elbo,
    flow_transformed,
    1000;
    adtype=ADTypes.AutoZygote(),
    optimizer=optimizer,
);

##
elbo_values = [s.elbo for s in stats]
window_size = 50
rolling_mean = [mean(elbo_values[max(1,i-window_size+1):i]) for i in 1:length(elbo_values)]

## 
samples = rand(q, 2000)
mean_samples = mean(samples, dims=2)
@test mean_samples[1] ≈ true_mu atol=0.25
@test mean_samples[2] ≈ true_sigma atol=0.25