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

function hopf_test()
    coll = CollocationSystem(hopf(), 1, 3)
    add_boundarycondition!(coll, (u₀, t₀, u₁, t₁, p) -> u₀)  # periodicity
    add_boundarycondition!(coll, (u₀, t₀, u₁, t₁, p) -> [0 ~ t₀, 0 ~ t₁ - 2π])  # start time and phase condition
    sol = t -> (cos(t), sin(t))
    ics = Dict(generate_initialconditions(coll, sol, (0, 2π)))
    fun = NonlinearFunction(NonlinearSystem(coll))

    res = fun(getindex.(Ref(ics), states(coll)), getindex.(Ref(ModelingToolkit.get_defaults(coll)), parameters(coll)))

    return (coll, ics, fun, res)
end

function test2()
    @variables x y
    sys = NonlinearSystem([x ~ y], [x, y], []; defaults=[y => 1])
    return NonlinearProblem(sys, [x => 0])
end
