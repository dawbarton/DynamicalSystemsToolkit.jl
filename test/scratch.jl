using DynamicalSystemsToolkit
using ModelingToolkit
using NonlinearSolve

function testsystem()
    @parameters t β σ
    @variables y₁(t) y₂(t)
    D = Differential(t)
    return ODESystem(
        [
            D(y₁) ~ β * y₁ - y₂ + σ * y₁ * (y₁^2 + y₂^2),
            D(y₂) ~ y₁ + β * y₂ + σ * y₂ * (y₁^2 + y₂^2),
        ],
        t,
        [y₁, y₂],
        [β, σ];
        defaults=[β => 1, σ => -1],
        name=:Hopf,
    )
end

coll = CollocationSystem(testsystem(), 5, 3)

function test2()
    @variables x y
    sys = NonlinearSystem([x ~ y], [x, y], []; defaults=[y => 1])
    return NonlinearProblem(sys, [x => 0])
end
