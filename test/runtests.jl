using DynamicalSystemsToolkit
using Test

@testset "Collocation" begin
    # Test the Chebyshev differentiation matrix implementation
    cheb = DynamicalSystemsToolkit.cheb
    x2 = [0, 1//2, 1]
    D2 = [-3 4 -1; -1 0 1; 1 -4 3]
    x3 = [0, 1//4, 3//4, 1]
    D3 = [-38//6 8 -8//3 1; -2 2//3 2 -2//3; 2//3 -2 -2//3 2; -1 8//3 -8 38//6]
    # x4 = [1, 1/sqrt(2), 0, -1/sqrt(2), -1] # could work out symbolically if want to test BigFloat accuracy
    @test all(cheb(Float64, 2) .≈ (x2, D2))
    @test all(cheb(Float64, 3) .≈ (x3, D3))
    @test all(eltype.(cheb(Float32, 3)) .=== (Float32, Float32))
end
