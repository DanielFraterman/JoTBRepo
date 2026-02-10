# Graphing p*phi for p=q for comparison to Rainey and Kerr's work
using Plots, Distributions, ProgressBars, LaTeXStrings
include("tau_leaping_alg.jl")

# Paramaters
C = 50000
k = 1.2

N = 101 # Number of grid points
nSamples = 10000 # Samples per mean

function keyQuantsEqual(N,nSamples,C,k)
    # Vector to store phi(p)
    Phi = Vector{Float64}(undef,N)

    # Vector to store samples 
    PhiSamples = Vector{Int64}(undef,nSamples)

    # Range of p=q to solve over
    ran = range(0,1,N)
    pbar = ProgressBar(total = N)

    # Loop over each p=q
    for i in 1:N
        # Sample phi nSamples times
        for l in 1:nSamples
            PhiSamples[l], _ = tau_leaping_alg(0,1,0,ran[i],ran[i],C,0.01,0.02,k) # Beta doesnt matter for this, setting tau=0.01
        end
        # Saving mean phi
        Phi[i] = 0.01*mean(PhiSamples)
        update(pbar)
    end
        
    # Plotting
    Plot1 = plot(ran,log.(Phi.*ran),legend=false)
    Plots.xlabel!(L"p=q"); Plots.ylabel!(L"\log(p\phi)")
    return Plot1
end

PQPlot = keyQuantsEqual(N,nSamples,C,k)
 


