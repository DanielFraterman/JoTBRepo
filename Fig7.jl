# Make plot on the gamma g axis showing either the final evo point for p or q

using Flux, Distributions, Plots, JLD2, LinearAlgebra, DifferentialEquations, ProgressBars, Suppressor, LaTeXStrings, Measures


# Parameters
k = 1.2
C = 50000
wMax = 2

# Getting model
phiScale = 2/(C*0.01)
name = "model_C"*string(C)*"_wMax"*string(wMax)*"_k"*replace(string(k), "." => "_")
folder = "NeuralNetwork/Models/"
model_state = JLD2.load(folder*name*".jld2", "model_state")
model = Chain(Dense(2 => 64,relu), Dense(64 => 64,relu), Dense(64 => 2,relu))
Flux.loadmodel!(model,model_state)
function PhiDelta(p,q)
    a = model([p,q])
    phi = a[1]/phiScale
    delta = a[2]
    return phi,delta
end

# Comp Model, for finding ybar an zbar
function dmdt(m,p,t) 
    #1 dydt = alpha - ksy -y sum(wi)
    #2 dsdt = k(1-p)sy - gamma*s + phi*delta*z
    #3 dzdt = gwn*(1-z) - delta*z
    #4 dw1dt = kpsy - (gamma+(1-q)y/1)w1
    #...
    #3+Wn dwndt = (1-q)yzw(n-1)/(n-1) - (gamma+(1-q)y/n+g(1-z))wn

    ddt = [p[1] - p[2]*m[1]*m[2] - m[1]*(sum(m[4:end])),
        p[2]*(1-p[3])*m[1]*m[2] - p[4]*m[2] + p[7]*p[8]*m[3],
        p[6]*m[end]*(1-m[3])-p[8]*m[3],
        p[2]*p[3]*m[1]*m[2] - (p[4] + (1-p[5])*m[1])*m[4]]
    for i in 2:(length(m[4:end])-1)
        append!(ddt,(1-p[5])*m[1]*m[2+i]/(i-1)-(p[4]+(1-p[5])*m[1]/i)*m[3+i])
    end
    append!(ddt,(1-p[5])*m[1]*m[end-1]/(p[9]-1)-(p[4]+(1-p[5])*m[1]/p[9]+p[6]*(1-m[3]))*m[end])

    return ddt
end


# simulates evolution paths through the p,q plane where step sizes and direction are randomly sampled from a normal Distribution
# finds the convergent stable point
function evolution(p::Float32,q::Float32,g,gamma,alpha,k,wMax,I)
    # Initialising vectors to store path of p and q
    P = Vector{Float64}(undef,I+1)
    Q = Vector{Float64}(undef,I+1)
    P[1] = p; Q[1] = q

    # Finding the value of phi and delta for resident p and q
    phi, delta = PhiDelta(p,q)

    # Finding steady state of the resident system
    par = [alpha,k,p,gamma,q,g,phi,delta,wMax]
    ode = SteadyStateProblem(dmdt,[0;1;0;zeros(Int64,wMax)],par)
    sol = solve(ode,DynamicSS(Rodas5()))
    ybar = sol.u[1]
    sbar = sol.u[2]
    zbar = sol.u[3]
    Wbar = sol.u[4:end]

    # Simulate I invasions
    for j in 1:I
        # Chooses a random step
        step = 0.001*randn(Float32) # the distance from resident pair the mutant will be
        ang = pi*rand(Float32) # the angle

        # Mutant traits
        pI = p + step*cos(ang)
        qI = q + step*sin(ang)
        
        # Ensuring p,q remain in unit square
        if pI > 1
            pI = 1.0f0
        elseif pI < 0
            pI = 0.0f0
        end
        if qI > 1
            qI = 1.0f0
        elseif qI < 0
            qI = 0.0f0
        end

        # Get phi and delta for mutants
        phiI, deltaI = PhiDelta(pI,qI)

        # Check for sucessful invasion
        nprod = prod(gamma .+ ybar*(1 -qI)./(1:(wMax-1)))
        INV = factorial(wMax-1)*(k*(1-pI)*ybar- gamma)*(gamma +(1 -qI) *ybar/wMax+g*(1-zbar))*nprod + k*pI*phiI*ybar^(wMax)*(1-qI)^(wMax-1)*g*(1-zbar) > 0 # Invasion condition
         
        # Runs if invasion successful
        if INV
            # Mutant traits replace residents'
            p = pI
            q = qI
            phi = phiI
            delta = deltaI
            par = [alpha,k,p,gamma,q,g,phi,delta,wMax]
            ode = SteadyStateProblem(dmdt,[0;1;0;zeros(Int64,wMax)],par)
            sol = solve(ode,DynamicSS(Rodas5()))
            ybar = sol.u[1]
            sbar = sol.u[2]
            zbar = sol.u[3]
            Wbar = sol.u[4:end]
        end

        # Records traits
        P[j+1] = p
        Q[j+1] = q
    end
    return P, Q
end


function css(alpha,wMax,k)
    I = 4000 # Invasion steps to perform
    n = 101 # number of points along each axis

    # Matricies to keep CSS values of p and q
    CSSP = Matrix{Float64}(undef,n,n)
    CSSQ = Matrix{Float64}(undef,n,n)

    # Range of g and gamma to solve over
    G = range(0,1,n)
    Gamma = range(0,1,n)

    pbar = ProgressBar(total = n^2)
    local P,Q

    # Loop over g
    for i in eachindex(G)
        # Loop over gamma
        for j in eachindex(Gamma)
            # Simulate evolution of p and q
            @suppress_err begin
             P,Q = evolution(0.5f0,0.4f0,G[i],Gamma[j],alpha,k,wMax,I)
            end
            # Taking the average of the last few trait pairs as due to approximation by the neural Network
            # and ode solver, traits move around CSS
            # invasion condition 
             CSSP[j,i] = mean(P[(I-100):I])
             CSSQ[j,i] = mean(Q[(I-100):I])
             update(pbar,1)
        end
    end

    # Plotting
    CustomGrad = cgrad([RGB(1,1,1),RGB(127/255,205/255,187/255),RGB(44/255,127/255,184/255)])

    CSSPlotP = Plots.contourf(G,Gamma,CSSP,color=CustomGrad)
    Plots.xlabel!(L"$g'$"); Plots.ylabel!(L"$\gamma'$"); Plots.title!(L"$p^*$")
    CSSPlotQ = Plots.contourf(G,Gamma,CSSQ,color=CustomGrad)
    Plots.xlabel!(L"$g'$"); Plots.ylabel!(L"$\gamma'$"); Plots.title!(L"$q^*$")
    return Plots.plot(CSSPlotP,CSSPlotQ)
end

# Plotting with alpha = 1
CSSPlotA1 = css(1,wMax,k)

# Plotting with alpha = 2
CSSPlotA3 = css(3,wMax,k)

# Assembelling plots
CSSPlot = plot(CSSPlotA1,CSSPlotA3,layout = (2,1))
labelPlot = plot(widen=false,axis = false,grid = false)
annotate!(0,1,L"(a)")
annotate!(0,0.38,L"(b)")
l = @layout [ a{0.0001w} [ b ; c ] ]
Fig7 = plot(labelPlot,CSSPlotA1,CSSPlotA3,layout=l,left_margin = [-5mm -1mm 2mm -1mm 2mm])