using PyTorchNormalizingFlows
using Documenter

DocMeta.setdocmeta!(PyTorchNormalizingFlows, :DocTestSetup, :(using PyTorchNormalizingFlows); recursive=true)

makedocs(;
    modules=[PyTorchNormalizingFlows],
    authors="Arnau Quera-Bofarull",
    sitename="PyTorchNormalizingFlows.jl",
    format=Documenter.HTML(;
        canonical="https://arnauqb.github.io/PyTorchNormalizingFlows.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/arnauqb/PyTorchNormalizingFlows.jl",
    devbranch="main",
)
