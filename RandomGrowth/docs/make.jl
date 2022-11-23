using RandomGrowth
using Documenter

DocMeta.setdocmeta!(RandomGrowth, :DocTestSetup, :(using RandomGrowth); recursive=true)

makedocs(;
    modules=[RandomGrowth],
    authors="Simeon David Schaub <schaub@mit.edu> and contributors",
    repo="https://github.com/simeonschaub/RandomGrowth.jl/blob/{commit}{path}#{line}",
    sitename="RandomGrowth.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://simeonschaub.github.io/RandomGrowth.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/simeonschaub/RandomGrowth.jl",
    devbranch="main",
)
