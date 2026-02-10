# Plots p*phi over p & q
using Plots, Distributions, ProgressBars, LaTeXStrings, Measures
include("tau_leaping_alg.jl")

N = 101 # Number of grid points per Axis
nSamples = 100 # Samples per mean

C = 50000 # Maximum size of mat
k = 1.2 # Additional growth parameter for S cells

function keyQuants(N,nSamples,C,k)

    # Initilising
    
    Samples = Vector{Int64}(undef,nSamples)
    SamplesLog = Vector{Int64}(undef,nSamples)
    Phi = Matrix{Float64}(undef,N,N)
    PhiLog = Matrix{Float64}(undef,N,N)
    pbar = ProgressBar(total = N^2)

    # For plotting on normal scale
    ran = range(0,1,N)
    
    # For plotting on log scale
    ranLog = 10 .^range(-3,-1,N)

    # Loop over p
    for i in 1:N
        # Loop over q
        for j in 1:N
            # Finding phi with nSamples
            for l in 1:nSamples
                Samples[l],_ = tau_leaping_alg(0,1,0,ran[j],ran[i],C,0.01,0.02,k) # Beta doesnt matter for this, setting tau=0.01
                SamplesLog[l],_ = tau_leaping_alg(0,1,0,ranLog[j],ranLog[i],C,0.01,0.02,k)
            end
            Phi[i,j] = 0.01*mean(Samples)
            PhiLog[i,j] = 0.01*mean(SamplesLog)
            update(pbar)
        end
    end
    Phi = Phi'
    PhiLog = PhiLog'

    CustomGrad = cgrad([RGB(1,1,1),RGB(127/255,205/255,187/255),RGB(44/255,127/255,184/255)])

    Plot0 = plot(layout=grid(1,4,widths=(94/200,6/200,94/200,6/200)),bottom_margin=5mm)
    tic1 = collect(range(extrema(Phi.*ran')..., 16))
    tic12 = tic1[2:end] .- (tic1[2]-tic1[1])/2
    contourf!(ran,ran,Phi.*ran',left_margin=5mm,levels=tic1,color=CustomGrad,colorbar=false,subplot=1)
    scatter!([1],[0.33],label=false,subplot = 1)
    xlabel!(L"p",subplot=1); ylabel!(L"q",subplot=1); title!(L"p\phi",subplot=1) 
    heatmap!([1], tic12, [tic12;;],ylims=[0,maximum(tic1)],right_margin=5mm,c=CustomGrad, colorbar=false, xaxis=false, ymirror=true,categorical = true, yticks=optimize_ticks(extrema(tic12)...;k_min=8,k_max=9)[1],tick_direction=:out,subplot=2)
    tic2 = collect(range(extrema(log10.(PhiLog.*ranLog'))..., 16))
    tic22 = tic2[2:end] .- (tic2[2]-tic2[1])/2
    lev = 10 .^tic22
    contourf!(ranLog,ranLog,PhiLog.*ranLog',left_margin=5mm,right_margin=5mm,levels=lev,axis=:log10,color=CustomGrad,colorbar=false,subplot=3)
    xlabel!(L"\log{(p)}",subplot=3); ylabel!(L"\log{(q)}",subplot=3); title!(L"p\phi",subplot=3) 
    heatmap!([1], lev, [tic22;;],c=CustomGrad,right_margin=5mm, colorbar=false, xaxis=false, ymirror=true,categorical = true,yaxis=:log10, yticks=10 .^optimize_ticks(extrema(tic22)...;k_min=8,k_max=9)[1],tick_direction=:out,subplot=4)
    contour!(ranLog,ranLog,PhiLog.*ranLog',clabels=true,levels=[0.2],subplot=3)

    Plot1 = plot(grid=false,axis=false)
    annotate!([0.19,0.75],[0,0],["(a)","(b)"])
    return plot(Plot0,Plot1,layout=grid(2,1,heights=(97/100,3/100)),size=(1200,412))
end

keyQuants(N,nSamples,C,k)
