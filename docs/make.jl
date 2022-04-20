using Dedisp
using Documenter

DocMeta.setdocmeta!(Dedisp, :DocTestSetup, :(using Dedisp); recursive=true)

makedocs(;
    modules=[Dedisp],
    authors="Kiran Shila <me@kiranshila.com> and contributors",
    repo="https://github.com/kiranshila/Dedisp.jl/blob/{commit}{path}#{line}",
    sitename="Dedisp.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kiranshila.github.io/Dedisp.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kiranshila/Dedisp.jl",
    devbranch="main",
)
