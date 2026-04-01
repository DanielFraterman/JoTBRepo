# Plotting high values of phi*p for different k values to showcase the bimodal surface around k=2.3
# ITs quicker to do this dimensionally than not
using Flux, Zygote, Plots, Optimization, OptimizationNLopt, OptimizationOptimJL, ProgressBars, LaTeXStrings, JLD2, Measures


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

# Making a function to get p*phi
function pPhi(x,pars)
    phi, _ = PhiDelta(x[1],x[2],pars[1])
    return x[1]*phi 
end

##

function phi_k(k)
    m = 101 # number of points on p q axes
    
    # Converting k to float32 for use with Flux
    k = Float32.(k)
    klen = length(k)

    # range of p and q to solve over
    pRan = Float32.(range(0.25,1.0,m))
    qRan = Float32.(range(0.0,0.5,m))

    # Vector to store values of phi
    phiVector = Vector{Matrix{Float32}}(undef,klen)

    # Vectors to store indexes and values of of (p^*, q^*)
    qStar = Vector{Int64}(undef,klen)
    pStar = Vector{Int64}(undef,klen)
    ESSP = Vector{Float32}(undef,klen)
    ESSQ = Vector{Float32}(undef,klen)

    # Loops over all values of k supplied
    for i in 1:klen
        # Finds p*phi
        phiVector[i] = pRan'.*getindex.(PhiDelta.(pRan',qRan,k[i]),1)
        
        # Gets index of (p^*,q^*)
        qStar[i], pStar[i] = argmax(phiVector[i]).I

        # Finds value of (p^*,q^*)
        ESSP[i] = pRan[pStar[i]]
        ESSQ[i] = qRan[qStar[i]]

        # Removes all p*phi values that are < 60 to highlight bimodal surface
        phiVector[i][phiVector[i] .< 60] .= NaN
    end

    # Defining custom colour gradient
    CustomGrad = cgrad([RGB(44/255,127/255,184/255),RGB(127/255,205/255,187/255),RGB(1,1,1)])

    # Creating a vector to hold each plot
    Plot = Vector(undef,klen)

    # Contour plots phi*p for each k value
    Plot[1] = Plots.contourf(pRan,qRan,phiVector[1],clims=(60,maximum(maximum.(phiVector))),cbar=false,color=CustomGrad)
    Plots.scatter!([ESSP[1]],[ESSQ[1]],label=L"(p^*,q^*)")
    kTitle = k[1]
    Plots.title!(L"k=%$kTitle")
    ylabel!(L"q")
    for i in 2:klen
        kTitle = k[i]
        Plot[i] = Plots.contourf(pRan,qRan,phiVector[i],clims=(60,maximum(maximum.(phiVector))),cbar=false,yaxis=false,color=CustomGrad)
        Plots.scatter!([ESSP[i]],[ESSQ[i]],label=L"(p^*,q^*)",labels=false)
        Plots.title!(L"k=%$kTitle")
    end

    # Combines all plots
    Plot1 = Plots.plot(Plot...,layout = (1,klen), right_margin=[-5mm -5mm -5mm 0mm], left_margin=[0mm -5mm -5mm -5mm])
    xlabel!(L"p")

    # Return plot
    return Plot1
end

phi_k([1.2,2.3,2.4,3.0])
