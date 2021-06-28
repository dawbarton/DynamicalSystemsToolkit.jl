using DynamicalSystemsToolkit
using ModelingToolkit
using NonlinearSolve

function hopf()
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
        name=:hopf,
    )
end

function hopf_test(; n_mesh=5, n_coll=4)
    coll = CollocationSystem(hopf(), n_mesh, n_coll)
    add_boundarycondition!(coll, (u₀, t₀, u₁, t₁, p) -> u₀ .- u₁)  # periodicity
    add_boundarycondition!(coll, (u₀, t₀, u₁, t₁, p) -> [0 ~ t₀, 0 ~ u₀[2]])  # start time and phase condition
    sol = t -> (cos(t), sin(t))
    ics = generate_initialconditions(coll, sol, (0, 2π))
    prob = NonlinearProblem(NonlinearSystem(coll), ics, [], jac=true)
    return (coll, prob)
end

function test2()
    @variables x y
    sys = NonlinearSystem([x ~ y], [x, y], []; defaults=[y => 1])
    return NonlinearProblem(sys, [x => 0])
end
