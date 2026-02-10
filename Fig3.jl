# Make plot on the k C axis showing either the final evo point for p or q
# It's quikcker to do this dimensionally than not
# Finds final spot by numerically maximising p*phi(p,q,k,C) rather than by simulating evo paths

using Distributions, Plots, Optimization, OptimizationNLopt, ProgressBars,LaTeXStrings, OptimizationOptimJL
include("tau_leaping_alg.jl")

# function to find p*phi through simulation
function pPhi(x,pars)
    # Initilising vector for samples
    PhiSamples = Vector{Int64}(undef,Int64(pars[3]))

    # Find phi
    for i in 1:Int64(pars[3])
        PhiSamples[i], _ = tau_leaping_alg(0,1,0,x[2],x[1],Int64(pars[2]),0.005,0.01,pars[1])
    end
    return -x[1]*0.01*mean(PhiSamples) # returns negative value so that minimisation alg can be used
end


function css(Samples)
    # Takes in "Samples" the number of simulations used to find phi
    
    n = 11 # number of points along each axis

    # Initilising matricies to hold ESS values
    ESSP = Matrix{Float64}(undef,n,n)
    ESSQ = Matrix{Float64}(undef,n,n)

    # Values of C and k to run over
    C = round.(Int64,10 .^range(3,18,n))
    K = range(1.0,10.0,n)

    pbar = ProgressBar(total = n^2)

    # Loop over C
    for i in eachindex(C)
        # Loop over k
        for j in eachindex(K)
            # parameters needed for pPhi function
            pars = [K[j],C[i],Samples]
            
            # Solving maximum of pPhi for unit square p and q
            f = OptimizationFunction(pPhi)
            prob = Optimization.OptimizationProblem(f, [0.5,0.5], pars, lb = [0.0, 0.0], ub = [1.0, 1.0])
            sol = solve(prob, NLopt.LN_NELDERMEAD()) 

            # Saves solution for plotting
            ESSP[j,i] = sol.u[1]
            ESSQ[j,i] = sol.u[2]
            update(pbar,1)
        end
    end

    CustomGrad = cgrad([RGB(1,1,1),RGB(127/255,205/255,187/255),RGB(44/255,127/255,184/255)])

    # Plot ESS value of p
    ESSPlotP = Plots.contourf(C,K,ESSP,color=CustomGrad, xaxis=:log10)
    Plots.xlabel!("C"); Plots.ylabel!("k"); Plots.title!(L"p^*")

    # Plot ESS value of q
    ESSPlotQ = Plots.contourf(C,K,ESSQ,color=CustomGrad, xaxis=:log10)
    Plots.xlabel!("C"); Plots.ylabel!("k"); Plots.title!(L"q^*")

    return Plots.plot(ESSPlotP,ESSPlotQ)
end

# Finds the css for each point using n samples
CSSplot = css(100)
