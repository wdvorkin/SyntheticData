using Plots
using JuMP, Mosek, MosekTools
using DataFrames, JSON, ArgParse
using Distributions, Random, StatsBase
using LinearAlgebra

function feature_trans(x,k,γ)
    n = length(x)
    X = zeros(n,k)  
    for i in 1:k
        μ = LinRange(minimum(x), maximum(x), k)[i]
        X[:,i] = φ.(x, μ, γ)
    end
    return X
end

function φ(x, μ, γ)
    ``` Gaussian radial basis function```
    return exp(-(γ*(μ-x))^2)
end

function get_records(data,set)
    Random.seed!(1)
    sample_ = sample(1:length(data["speed"]),set[:N],replace=true)
    Random.seed!()
    return Dict("speed" => data["speed"][sample_], "power" => data["power"][sample_])
end

function fit_model(set,real_data)
    β = (real_data[:X]'*real_data[:X] + set[:λ] * I)^(-1)*real_data[:X]'*real_data[:power]

    loss = norm(real_data[:X]*β .- real_data[:power],2).^2

    return Dict(:β => β, :loss => loss)
end

function basis_vector_alpha(n,i,α) 
    I = zeros(n); I[i] = α
    return I
end

function sensitivities(real_data)
    # regression weights 
    δβ = opnorm((real_data[:X]'*real_data[:X] + set[:λ] * I)^(-1)*real_data[:X]',1)*set[:α]
    # regression loss 
    n = length(real_data[:power])
    δl = zeros(n)
    for j in 1:n
        δl[j] = norm((real_data[:X]*(real_data[:X]'*real_data[:X] + set[:λ] * I)^(-1)*real_data[:X]' - I)*basis_vector_alpha(n,j,set[:α]),2)^2
    end
    δl = maximum(δl)
    return Dict(:δβ => δβ, :δl => δl)
end

function post_processing(l̅,β̅,y,X)
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))

    @variable(model, ỹ[1:set[:N]])
    @variable(model, t_l)
    @variable(model, t_y)
    @variable(model, t_β)
    @variable(model, β[1:set[:k]])
    @variable(model, l)

    @objective(model, Min, t_l + 1e-5*t_β + 1e-5*t_y)
    @constraint(model, [t_l; l-l̅] in SecondOrderCone())
    @constraint(model, [t_β; β-β̅] in SecondOrderCone())
    @constraint(model, [t_y; ỹ-y] in SecondOrderCone())

    @constraint(model, ỹ .<= 1)
    @constraint(model, ỹ .>= 0)

    @constraint(model, β .== (X'*X + set[:λ] * I)^(-1)*X'*ỹ)
    @constraint(model, [0.5;l; X*β - ỹ] in RotatedSecondOrderCone())

    optimize!(model)
    status = "$(termination_status(model))"
    @info("post-processing optimziation terminates with status: $(status)")

    return Dict(:ỹ => JuMP.value.(ỹ), :t_l => JuMP.value.(t_l), :t_y => JuMP.value.(t_y), :β => JuMP.value.(β), :CPUtime => [solve_time(model)])
end


function Lap_mechanism(real_data,set)
    data = deepcopy(real_data)
    scale = set[:α]/(set[:ε])
    ỹ_0 = real_data[:power] .+ rand(Laplace(0,scale),length(real_data[:power]))
    data[:power] = min.(max.(ỹ_0,0),1)  
    return data
end

function add_synt_data_plot(plo,ỹ,col,title_)
    plo_ = plot(plo)
    scatter!(real_data[:speed],ỹ,c=col,alpha=0.2,markersize = 2)
    plot!(title=title_)
    return plo_
end

function WPO_algorithm(real_data,set)

    # compute model sensitivities 
    δ = sensitivities(real_data)

    # Step 1: initialize synthetic data 
    scale = set[:α]/(set[:ε]/3)
    ỹ_0 = real_data[:power] .+ rand(Laplace(0,scale),length(real_data[:power]))

    # Step 2: Laplace mechanism to privately estimate regression 
    scale = δ[:δl]/(set[:ε]/3)
    l̅ = sol_reg[:loss] .+ rand(Laplace(0,scale))
    scale = δ[:δβ]/(set[:ε]/3)
    β̅ = sol_reg[:β] .+ rand(Laplace(0,scale))

    # Step 3: post-processing of synthetic power records
    sol_pp = post_processing(l̅,β̅,ỹ_0,real_data[:X])

    # return syntetic data
    synt_data = deepcopy(real_data)
    synt_data[:power] = min.(max.(sol_pp[:ỹ],0),1) 
    merge!(synt_data,Dict(:initial => ỹ_0))
    merge!(synt_data,Dict(:CPUtime => sol_pp[:CPUtime]))

    return synt_data
end

# upload data
cd(dirname(@__FILE__))
wind_data = JSON.parsefile("data/power_curve_data.json")
wind_data_keys = collect(keys(wind_data))

# pick one turbine
tubine = "GE.2.75.103"
data = wind_data[tubine]

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--adjacency", "-a"
            help = "adjacency datasets in percentage"
            arg_type = Float64
            default = 15.0
        "--loss", "-e"
            help = "privacy loss ε"
            arg_type = Float64
            default = 1.0
    end
    return parse_args(s)
end
args = parse_commandline()

# experiment settings 
set = Dict(:k => 5, :γ => 0.5, :λ => 0.001, :α => args["adjacency"]/100, :ε => args["loss"], :N => 1000)

# pick a random sample of N records 
wp_records = get_records(data,set)
real_data = Dict(
    :speed => wp_records["speed"],
    :power => wp_records["power"],
    :X => feature_trans(wp_records["speed"],set[:k],set[:γ])
)

# solve regression problem
sol_reg = fit_model(set,real_data)

# make a plot of data and regression population
plo = plot(
    xlims=(minimum(real_data[:speed]),maximum(real_data[:speed])),
    ylims=(-.02,1.02), 
    frame=:box,legend=false, xlabel="wind speed (m/s)",ylabel="power output",  xtickfontsize=14,ytickfontsize=14, labelfontsize=14)
# data plot
scatter!(real_data[:speed],real_data[:power],c=:black,markersize = 2)
# regression line
x_range = collect(LinRange(minimum(real_data[:speed]),maximum(real_data[:speed]),100))
plot!(x_range,feature_trans(x_range,set[:k],set[:γ])*sol_reg[:β],lw=3,alpha=0.5,c=:blue)

# get syntehtic data using WPO Algorithm
synt_data_wpo = WPO_algorithm(real_data,set)
# get syntehtic data using Laplace mechanism
synt_data_lap = Lap_mechanism(real_data,set)

# plot syntehtic data (WPO)
plo_wpo = add_synt_data_plot(plo,synt_data_wpo[:power],:green,"WPO algorithm")
# plot syntehtic data (Laplace)
plo_lap = add_synt_data_plot(plo,synt_data_lap[:power],:red,"Laplace mechanism")

# compare results 
compare_plot = plot(plo_lap,plo_wpo,size=(1000,300))
plot!(bottom_margin=10Plots.mm)
plot!(left_margin=5Plots.mm)
savefig(compare_plot, "Lap_WPO_comparison.pdf")
display(compare_plot)

# compute regression loss on datasets
sol_real = fit_model(set,real_data)
sol_wpo_ = fit_model(set,synt_data_wpo)
sol_lap_ = fit_model(set,synt_data_lap)
println("regression loss:")
println("on real data: $(sol_real[:loss])")
println("on synthetic data (WPO): $(sol_wpo_[:loss])")
println("on synthetic data (Lap): $(sol_lap_[:loss])")