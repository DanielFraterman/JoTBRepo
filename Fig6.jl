# Finds the expected mutation paths by finding the directin that the invasion function increases in
## Generates the expected path for the comp model
using Flux, Distributions, Plots, JLD2, LinearAlgebra, DifferentialEquations, SciMLSensitivity

# Mat Parameters
k = 1.2
C = 50000
wMax = 2

# More pond parameters
g = 0.05
gamma = 0.01
alpha = 1

# Getting model
phiScale = 2/(C*0.01)
name = "model_C"*string(C)*"_wMax"*string(wMax)*"_k"*replace(string(k), "." => "_")
folder = "NeuralNetwork/Models/"
model_state = JLD2.load(folder*name*".jld2", "model_state")
model = Chain(Dense(2 => 64,relu), Dense(64 => 64,relu), Dense(64 => 2,relu))
Flux.loadmodel!(model,model_state)
function PhiDelta(p,q) # Function that rescales to output of the model back to normal
    a = model([p,q])
    phi = a[1]/phiScale
    delta = a[2]
    return phi,delta
end

# Comp Model, for finding ybar and zbar numerically
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



## Making vector field

# Checking value of invasion condition
function InvFunc(p,q,ybar,zbar,g,gamma) # successful inv if >0
    phi = getindex.(PhiDelta.(p,q),1)
    qybar = (1 .-q)*ybar
    Inv = g*k*(1-zbar)*ybar*phi.*p.*qybar.^(wMax-1) - factorial(wMax-1)*(gamma .- k*ybar*(1 .-p)).*(gamma .+ qybar/wMax.+g*(1-zbar)).*vec(prod(gamma.+qybar./((1:(wMax-1))'),dims=2))

    return Inv
end

n = 21 # number of grid points for p and q
p = Float32.(ones(n)*range(0,1,n)'); q = Float32.(range(0,1,n)*ones(n)')

# Vector to hold gradient of invasion condition
grads = Vector{Any}(undef,n^2)

# Find gradient for each pair of p and q
for i in eachindex(p)
    # Getting resident phi and delta
    (phiR,deltaR) = PhiDelta(p[i],q[i])

    # Parameters for ODE model
    par = [alpha,k,p[i],gamma,q[i],g,phiR,deltaR,wMax]
    ode = SteadyStateProblem(dmdt,[0;10;0;zeros(Int64,wMax)],par)
    # Solving ODE model
    sol = solve(ode,DynamicSS(Rodas5()))
    ybar = sol.u[1]
    zbar = sol.u[3]

    # Getting gradient
    grads[i] = gradient((x,y)->InvFunc([x],[y],ybar,zbar,g,gamma)[1],p[i],q[i]) # Finding gradient
end

# Plotting vector field
Plot = Plots.plot()
Plots.quiver!(p[:],q[:],quiver=(0.00001*getindex.(grads,1),0.00001*getindex.(grads,2)),c=:gray)
Plots.xlabel!("p"); Plots.ylabel!("q"); Plots.xlims!(0,1); Plots.ylims!(0,1)

## Plotting sample paths on top

# Simulating evo paths
function evolution(p::Float32,q::Float32,g,gamma,alpha,k,wMax,I)
    # Vectors to store the path over p and q
    P = Vector{Float32}(undef,I+1)
    Q = Vector{Float32}(undef,I+1)
    P[1] = p; Q[1] = q

    # Getting phi and delta for current residents
    phi, delta = PhiDelta(p,q)

    # Solving ODE model
    par = [alpha,k,p,gamma,q,g,phi,delta,wMax]
    ode = SteadyStateProblem(dmdt,[0;10;0;zeros(Int64,wMax)],par)
    sol = solve(ode,DynamicSS(Rodas5()))
    ybar = sol.u[1]
    sbar = sol.u[2]
    zbar = sol.u[3]
    Wbar = sol.u[4:end]

    # Simulating I Invasion attemps
    for j in 1:I
        step = 0.005*randn(Float32) # the distance from resident pair the mutant will be
        ang = pi*rand(Float32) # the angle

        # Mutant traits
        pI = p + step*cos(ang)
        qI = q + step*sin(ang)

        # Ensuring traits stay in unit square
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
        INV = factorial(wMax-1)*(k*(1-pI)*ybar- gamma)*(gamma +(1 -qI) *ybar/wMax+g*(1-zbar))*nprod + k*pI*phiI*ybar^(wMax)*(1-qI).^(wMax-1)*g*(1-zbar) > 0 # Invasion condition 

        # Runs if invasion is successful
        if INV
            # Mutants replace residents
            p = pI
            q = qI
            phi = phiI
            delta = deltaI
            par = [alpha,k,p,gamma,q,g,phi,delta,wMax]
            ode = SteadyStateProblem(dmdt,[0;10;0;zeros(Int64,wMax)],par)
            sol = solve(ode,DynamicSS(Rodas5()))
            ybar = sol.u[1]
            sbar = sol.u[2]
            zbar = sol.u[3]
            Wbar = sol.u[4:end]
        end
        # Stores the current resident traits
        P[j+1] = p
        Q[j+1] = q
    end
    return P, Q
end

I = 3000 # Number of attempted invasions

# Initial traits to start from
P0 = Float32.([0.0,0.5,0.25,0.8,0.9,0.4]); Q0 = Float32.([0.0,0.4,0.35,0.05,0.9,0.1]) # Initial (p,q) to use
P = Vector{Vector{Float32}}(undef,length(P0))
Q = Vector{Vector{Float32}}(undef,length(Q0))

# Simulating evolution
for i in eachindex(P0)
    P[i], Q[i] = evolution(P0[i],Q0[i],g,gamma,alpha,k,wMax,I)
end

# Plotting ontop of the vector field
Plot2 = plot(Plot)
for i in eachindex(P0)
    plot!(P[i],Q[i],label=false,lw=2)
end
Plot2
scatter!(getindex.(P,1),getindex.(Q,1),label="Initial point") # Plotting start points
scatter!(getindex.(P,I+1),getindex.(Q,I+1),label="End point") # Plotting end points
# savefig("GradientPath_k1_2_g0_05_gamma0_01_alpha1.pdf")