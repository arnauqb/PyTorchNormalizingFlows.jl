using PyTorchNormalizingFlows
using DynamicPPL
using Test
using Distributions
using Zygote

@testset "PyTorchNormalizingFlows.jl" begin
    @testset "Flows" begin
        @testset "RealNVP" begin
            flow = make_real_nvp_flow_torch(2, 2, 16)
            @test size(rand(flow)) == (2,)
            @test size(logpdf(flow, rand(flow))) == ()
        end
        @testset "MaskedAffineAutoregressive" begin
            flow = make_masked_affine_autoregressive_flow_torch(dim=2, n_layers=2, n_units=16)
            @test size(rand(flow)) == (2,)
            @test size(logpdf(flow, rand(flow))) == ()
        end
        @testset "NeuralSpline" begin
            flow = make_neural_spline_flow_torch(2, 2, 16)
            @test size(rand(flow)) == (2,)
            @test size(logpdf(flow, rand(flow))) == ()
        end
        @testset "Planar" begin
            flow = make_planar_flow_torch(2, 2)
            @test size(rand(flow)) == (2,)
            @test size(logpdf(flow, rand(flow))) == ()
        end
    end
    @testset "Zygote rules" begin
        @testset "Planar" begin
            flow = make_planar_flow_torch(2, 2)
            grad = Zygote.gradient(logpdf, flow, rand(flow))
            @test !any(isnan.(grad[1].params_flat))
            @test all(grad[1].params_flat .!= 0)
        end
        @testset "RealNVP" begin
            flow = make_real_nvp_flow_torch(2, 2, 16)
            grad = Zygote.gradient(logpdf, flow, rand(flow))
            @test !any(isnan.(grad[1].params_flat))
            @test any(grad[1].params_flat .!= 0)
        end
        @testset "MaskedAffineAutoregressive" begin
            flow = make_masked_affine_autoregressive_flow_torch(dim=2, n_layers=2, n_units=16)
            grad = Zygote.gradient(logpdf, flow, rand(flow))
            @test !any(isnan.(grad[1].params_flat))
            @test any(grad[1].params_flat .!= 0)
        end
        @testset "NeuralSpline" begin
            flow = make_neural_spline_flow_torch(2, 2, 16)
            grad = Zygote.gradient(logpdf, flow, rand(flow))
            @test !any(isnan.(grad[1].params_flat))
            @test any(grad[1].params_flat .!= 0)
        end
    end
    @testset "Integration with AdvancedVI" begin
        include("vi_tests.jl")
    end
end
