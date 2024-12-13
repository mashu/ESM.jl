using ESM
using Documenter

DocMeta.setdocmeta!(ESM, :DocTestSetup, :(using ESM); recursive=true)

makedocs(;
    modules=[ESM],
    authors="Mateusz Kaduk <mateusz.kaduk@gmail.com> and contributors",
    sitename="ESM.jl",
    format=Documenter.HTML(;
        canonical="https://mashu.github.io/ESM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mashu/ESM.jl",
    devbranch="main",
)
