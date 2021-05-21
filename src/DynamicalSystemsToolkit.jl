module DynamicalSystemsToolkit

using ModelingToolkit
using ModelingToolkit: get_iv, operation
using RuntimeGeneratedFunctions
using DocStringExtensions
using LinearAlgebra

RuntimeGeneratedFunctions.init(@__MODULE__)

export collocation

include("collocation.jl")

end
