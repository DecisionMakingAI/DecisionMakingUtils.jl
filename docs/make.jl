using Documenter
using DecisionMakingUtils

DocMeta.setdocmeta!(DecisionMakingUtils, :DocTestSetup, :(using DecisionMakingUtils); recursive=true)
makedocs(
    sitename = "DecisionMakingUtils",
    format = Documenter.HTML(),
    modules = [DecisionMakingUtils]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
