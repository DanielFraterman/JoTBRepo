include("../tau_leaping_alg.jl")
using ProgressBars, Distributions, Plots, Flux, LinearAlgebra, Printf, JLD2, LaTeXStrings
# Used to train neural netwroks taking inputs p and q with outputs phi and delta


## Data collection

# Parameters
k = 1.2
wMax = 2
C = 50000
tau = 0.02 # Equivalant to tau=0.01, beta=2

# Collecting data
n = 1000000 # data points
nM = 1000 # samples per mean
p = rand(n)
q = rand(n)


# Finding phi and delta for each p and q
Phi = Vector{Float32}(undef,nM); Delta = Vector{Float32}(undef,nM)
phi = Vector{Float32}(undef,n)
delta = Vector{Float32}(undef,n)
pbar = ProgressBar(total = n)
for i in 1:n
    for j in 1:nM
        Phi[j], Delta[j] = tau_leaping_alg(0,wMax,0,q[i],p[i],C,tau,tau,k)
    end
    phi[i] = 0.01*mean(Phi)
    delta[i] = 1/mean(Delta)
    update(pbar,1)
end

# Flux requires Float32's
p = Float32.(p); q = Float32.(q)  

# Scaling output
phiScale = 2/(C*0.01) # phiScale*phi: phi -> [0,1]

# Formatting input/output for training/testing
X = vcat(p',q') 
Y = vcat(phiScale*phi',delta')
trainingBatched = Flux.DataLoader((X[:,1:(floor(Int64,0.9*n))],Y[:,1:(floor(Int64,0.9*n))]),batchsize = 512)
testingX = X[:,(floor(Int64,0.9*n)+1):end]
testingY = Y[:,(floor(Int64,0.9*n)+1):end]

## Models

# Number of nodes and layers to sweep over
Nodes = [16,32,64,128,256]
HiddenLayers = [1,2,3,4]

# Initialising matrix to store each model and their losses
Models = Matrix(undef,length(HiddenLayers),length(Nodes))
TestLoss = Matrix(undef,length(HiddenLayers),length(Nodes))
Figs = Matrix(undef,length(HiddenLayers),length(Nodes))

# Loop over Hidden Layers
for i in eachindex(HiddenLayers)
    # Loop over Nodes
    for j in eachindex(Nodes)

        # Creating model given arcitecture
        if HiddenLayers[i] == 1
            Models[i,j] = Chain(Dense(2 => Nodes[j],relu), Dense(Nodes[j] => 2,relu))
        elseif HiddenLayers[i] == 2
            Models[i,j] = Chain(Dense(2 => Nodes[j],relu), Dense(Nodes[j] => Nodes[j],relu), Dense(Nodes[j] => 2,relu))
        elseif HiddenLayers[i] == 3
            Models[i,j] = Chain(Dense(2 => Nodes[j],relu), Dense(Nodes[j] => Nodes[j],relu), Dense(Nodes[j] => Nodes[j],relu), Dense(Nodes[j] => 2,relu))
        elseif HiddenLayers[i] == 4
            Models[i,j] = Chain(Dense(2 => Nodes[j],relu), Dense(Nodes[j] => Nodes[j],relu), Dense(Nodes[j] => Nodes[j],relu), Dense(Nodes[j] => Nodes[j],relu), Dense(Nodes[j] => 2,relu))
        end
    end
end

## Training

# Number of epochs
epochs = 2000
pbar = ProgressBar(total=epochs)

for i in eachindex(HiddenLayers)
    for j in eachindex(Nodes)

        model = Models[i,j]

        # Optimisation algorithm to use
        opt = Flux.setup(Adam(0.001), model)

        # Loss function to use
        loss(m,x,y) = Flux.Losses.mse(m(x),y)

        # Training model
        for epoch in 1:epochs
            for (x,y) in trainingBatched
                grads = gradient(m->loss(m,x,y),model)
                Flux.update!(opt, model, grads[1])
            end
            update(pbar,1)
        end

        # Plotting results
        CustomGrad = cgrad([RGB(1,1,1),RGB(127/255,205/255,187/255),RGB(44/255,127/255,184/255)])
        TestLoss[i,j] = loss(model,testingX,testingY)
        info ="Hidden Layers: "*string(HiddenLayers[i])*", Nodes: "*string(Nodes[j])*", Loss: "*@sprintf "%.2E" TestLoss[i,j]
        P1 = contourf(range(0f0,1f0,101),range(0f0,1f0,101),(x,y)->model([x,y])[1]./phiScale, title=L"\phi",color=CustomGrad); xlabel!("p"); ylabel!("q")
        P2 = contourf(range(0f0,1f0,101),range(0f0,1f0,101),(x,y)->model([x,y])[2], title=L"\delta",color=CustomGrad); xlabel!("p"); ylabel!("q")
        P3 = plot(grid=false,showaxis=false); annotate!(0.5,0.5,info)
        P0 = plot(P1,P2)
        Figs[i,j] = plot(P3,P0,layout=grid(2,1,heights=[0.1,0.9]))
    end
end


## Saving model

# Node and layers id wanting to be saves
nodeID = 3; hiddenlayerID = 2

# saving using jld2
model_state = Flux.state(Models[hiddenlayerID,nodeID])
name = "model_C"*string(C)*"_wMax"*string(wMax)*"_k"*replace(string(k), "." => "_")
folder = "NeuralNetwork/Models/"
jldsave(folder*name*".jld2"; model_state)
savefig(Figs[hiddenlayerID,nodeID],folder*name*".pdf")


# Loading Model

# model_state = JLD2.load(folder*name*".jld2", "model_state")
# model =  use same arcitecture as the model being loaded
# Flux.loadmodel!(model,model_state)