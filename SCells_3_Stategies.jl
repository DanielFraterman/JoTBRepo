using Random, Distributions, LaTeXStrings, Plots
include("tau_leaping_alg.jl")

q = [0.33,0.04,0.03] # Mutation rate from W to S cells
p = [1,0.39,0.51] # Mutation rate from S to W cells
k = [1.2,3.0,10.0] # where k*beta is reproduction rate of S cells

n = 1000 # number of samples

function mat_samples(n,p,q,k) # Returns vector of scells at death and time of death 
    Scells = Vector{Int64}(undef,n)
    Time = Vector{Float64}(undef,n)

    for l in 1:n
        Scells[l], Time[l] = tau_leaping_alg(0,1,0,q,p,50000,0.01,0.02,k) # tau=0.01, beta=2
    end
    
    return Time, Scells
end


# Finding scells at death and time of death for each strategy and value of k
Times = Matrix{Vector{Float64}}(undef,length(q),length(k))
Scells = Matrix{Vector{Int64}}(undef,length(q),length(k))
for i in eachindex(q)
    for j in eachindex(k)
        Times[i,j], Scells[i,j] = mat_samples(n,p[i],q[i],k[j])
    end
end


# Plotting
ScatterComp = plot(layout=(1,length(k)))
y = [0.5,1.0,1.5,2.0,2.5] * 10^4
for i in eachindex(q)
    Q = q[i]; P = p[i]
    scatter!(Times[i,:],Scells[i,:],labels=[L"k=1.2" L"k=3.0" L"k=10.0"],alpha=0.1,markersize=2,markerstrokewidth=0,xlims=[0,10],ylims=[0,25000],yaxis=(formatter=y->string(round(y / 10^4,sigdigits=2))),subplot=i)
    plot!(title=latexstring("\$q=$(Q)\$, \$p=$(P)\$"),subplot=i)
    if i == 1
        annotate!([(-1, maximum(y) * 1.05, Plots.text(L"\times10^{4}", 11, :black, :center))],subplot=1)
    else
        plot!(legend=false,subplot=i)
    end
end
ScatterComp
xlabel!("Time of mat death"); ylabel!("Number of S cells",subplot=1)


# Plotting as histogram
ScatterComp = plot(layout=(1,length(k)))
y = [0.5,1.0,1.5,2.0,2.5] * 10^4
for i in eachindex(q)
    Q = q[i]; P = p[i]
    scatter!(Times[i,:],Scells[i,:],labels=[L"k=1.2" L"k=3.0" L"k=10.0"],markersize=2,markerstrokewidth=0,xlims=[0,10],ylims=[0,25000],yaxis=(formatter=y->string(round(y / 10^4,sigdigits=2))),alpha = 0.1,subplot=i)
    plot!(title=latexstring("\$q=$(Q)\$, \$p=$(P)\$"),subplot=i)
    if i == 1
        annotate!([(-1, maximum(y) * 1.05, Plots.text(L"\times10^{4}", 11, :black, :center))],subplot=1)
    else
        plot!(legend=false,subplot=i)
    end
end
ScatterComp
xlabel!("Time of mat death"); ylabel!("Number of S cells",subplot=1)
