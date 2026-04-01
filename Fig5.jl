# Find the average cumulative number of cells produced by W and S cells 
# under the two different strategies
using Flux, Optimization, OptimizationNLopt, JLD2, PoissonRandom, Plots, LaTeXStrings

# Mat Parameters
C = 50000
wMax = 1
kupper = 5
klower = 1

# Getting model
phiScale = 2/(C*0.01)
name = "modelK"*string(klower)*"-"*string(kupper)*"_C"*string(C)*"_wMax"*string(wMax)
folder = "NeuralNetwork/Models/"
model_state = JLD2.load(folder*name*".jld2", "model_state")
model = Chain(Dense(3 => 128,relu), Dense(128 => 128,relu), Dense(128 => 128,relu), Dense(128 => 2,relu))
Flux.loadmodel!(model,model_state)
function PhiDelta(p,q,k) # Function that rescales to output of the model back to normal
    a = model([p,q,(k.-klower)/(kupper-klower)])
    phi = a[1]/phiScale
    delta = a[2]
    return phi,delta
end

# Value of k to use
k = [1.2f0,4.0f0]

## Finding the strategy
function fun(m,p)
    phi, _ = PhiDelta(Float32(m[1]),Float32(m[2]),Float32(p[1]))
    return -phi*m[1]
end

# Vector to hold p* and q* values
pStar = Vector{Float64}(undef,length(k)); qStar = Vector{Float64}(undef,length(k))
for i in eachindex(k)
    f = OptimizationFunction(fun)
    prob = Optimization.OptimizationProblem(f, [0.5f0,0.5f0], [k[i]], lb = [0.0f0, 0.0f0], ub = [1.0f0, 1.0f0])
    sol = solve(prob, NLopt.LN_NELDERMEAD()) 
    pStar[i] = sol.u[1]; qStar[i] = sol.u[2]
end



## Simulating mat

function tau_leaping_birth_track(C,k,p,q,tau,Beta)
    # Vector where xBirthy holds how many y's have been born by x's
    wBirthw = [0]; wBirths = [0]
    sBirthw = [0]; sBirths = [0]

    # Initial values
    S = 0; W = 1; N = 0; tauBeta= tau*Beta
    while S < W && S + W < C # Runs whilst max size hasnt been met and W>S
        pks = p*k*S
        qw = q*W
        # Births from W cells
        append!(wBirthw,pois_rand(tauBeta*(W-qw)))
        append!(wBirths,pois_rand(tauBeta*(qw)))
        # Births from S Cells
        append!(sBirthw,pois_rand(tauBeta*(pks)))
        append!(sBirths,pois_rand(tauBeta*(k*S-pks)))

        # Add births to population
        W += wBirthw[end] + sBirthw[end]
        S += wBirths[end] + sBirths[end]

        N += 1 # Increments timestep
    end
    return wBirthw, wBirths, sBirthw, sBirths, N*tau
end

tau = 0.01; beta = 2

wBw, wBs, sBw, sBs = [Vector{Vector{Int64}}(undef,length(k)) for i in 1:4]
time = Vector{Float64}(undef,length(k))
for i in eachindex(k)
    wBw[i], wBs[i], sBw[i], sBs[i], time[i] = tau_leaping_birth_track(C,k[i],pStar[i],qStar[i],tau,beta)
end

## Cumulative plot
CumPlot = plot(yaxis=(:log10, [0.9, :auto]))
xlabel!("Time"); ylabel!("Number of cells")
for i in eachindex(k)
    ps = round(pStar[i],digits=2); qs = round(qStar[i],digits=2)
    plot!(0:0.01:time[i],cumsum(sBw[i]+sBs[i]),color=i,linestyle=:dash,label="")
    plot!(0:0.01:time[i],cumsum(wBw[i]+wBs[i]),color=i,label="")
    plot!([1], [0], label = latexstring("Strategy \$(p,q)=($ps,$qs)\$"), color =i,linewidth=5)
end
CumPlot
plot!([1], [0], linestyle = :dash, label = "Daughters of SM cells", color = "black")
plot!([1], [0], label = "Daughters of WS cells", color = "black")

## Pie chart of mat makeup at end of life

PiePlots = Vector{Any}(undef,length(k))
for i in eachindex(k)
    PiePlots[i] = plot(title="k="*string(k[i]),framestyle = :box,ticks=false,grid=false,title_location=:left)
    pie!(["WS cells","SM cells"],[wBw[i][end]+sBw[i][end],wBs[i][end]+sBs[i][end]],inset =bbox(0.07,0.16,0.43,0.78),seriescolor = ["#FDBF03","#FF0000"],legend=:topright,legendfont=font(7),subplot=2)
    pie!(["Cells produced by WS cells","Cells produced by SM cells"],[wBs[i][end]+wBw[i][end],sBw[i][end]+sBs[i][end]],inset =bbox(0.5,0.16,0.43,0.78),legend=:topright,seriescolor = ["#FDBF03","#FF0000"],legendfont=font(7),subplot=3)
end
PiePlot = plot(PiePlots...,layout = (2,1))
