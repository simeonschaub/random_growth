using RandomGrowth
using Documenter

DocMeta.setdocmeta!(RandomGrowth, :DocTestSetup, :(using RandomGrowth); recursive=true)

makedocs(;
    modules=[RandomGrowth],
    authors="Simeon David Schaub <simeondavidschaub99@gmail.com> and contributors",
    repo="https://github.com/simeonschaub/random_growth/blob/{commit}/RandomGrowth{path}#{line}",
    sitename="RandomGrowth.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://simeonschaub.github.io/random_growth",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/simeonschaub/random_growth",
    devbranch="main",
    dirname="RandomGrowth.jl",
)
