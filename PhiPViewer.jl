# This creates an intereactable widget, displaying the surfaces of phi and phi*P
# as k is varied. This allows us to see what causes the trade off from p -> phi to become
# viable. Additionally, we see the value of phi and phi*p evaluated at
# q^*. q^* maximises both surfaces for a given p. This allows for a more intuitive
# understanding of why the trade off mentioned is viable for large k.
# As k is increased, SM cells grow so fast that small variations in p from the value that maximises phi
# cause phi to drastical decrease, much faster than when k is small. Thus, p=1 significantly decreases phi and thus phi*p.
# In comparison, when k is small, phi is only slightly reduced for p=1. Additionally, when k is increased,
# the value of p that maximises phi is increased, meaning p need be traded less in order to maximise phi.
using Flux, Plots,  LaTeXStrings, JLD2, GLMakie, CairoMakie

# Mat Parameters
# Maximum mat size
C = 50000
# Number of cells that seed a mat
wMax = 1
# Upper and lower bounds on k (NN was trained with 1<=k<=5)
kupper = 5
klower = 1

# Getting model
phiScale = 2/(C*0.01)
name = "modelK"*string(klower)*"-"*string(kupper)*"_C"*string(C)*"_wMax"*string(wMax)
folder = "NeuralNetwork/Models/"
model_state = JLD2.load(folder*name*".jld2", "model_state")
model = Chain(Dense(3 => 128,relu), Dense(128 => 128,relu), Dense(128 => 128,relu), Dense(128 => 2,relu))
Flux.loadmodel!(model,model_state)
function Phi(p,q,k) # Function that rescales to output of the model back to normal
    a = model([p,q,(k.-klower)/(kupper-klower)])
    phi = a[1]/phiScale
    return phi
end

# Values of k on the interactable slider
K = Float32.(range(1,5,1001))

# Initiate the widget
figpphip = GLMakie.Figure(size = (600,600))

# Adding the slider to the widget
sg = SliderGrid(
    figpphip[3, 1:2],
    (label = "k", range = K, format = "{:.3f}", startvalue = 1.2f0),
    width = 400,
    height = 100,
    tellheight = false)

sliderobservables = [s.value for s in sg.sliders] # Getting the value from the slider
vals = lift(sliderobservables...) do slvalues...
    [slvalues...]
end

# Function to calculate phi and phi*p
function get_fig(vals)
    # Number of points for p and q
    n = 101
    pm = Float32.(ones(n)*range(0,1,n)'); qm = Float32.(range(0,1,n)*ones(n)')

    # Getting phi for each point
    phi = Phi.(pm,qm,vals[1])

    # Getting phi for q=q^*
    phiQstar = maximum(phi,dims=1)[1,:]

    # Getting phi*p for q=q^*
    phipQstar = phiQstar .* range(0,1,n)

    # Removing small values from phi and phi*p such that behaviour need local maximums is clearer
    phip = phi'.*pm'
    phip[phip .< 60] .= NaN
    phi[phi .< 60] .= NaN

    return [phi',phip,phiQstar,phipQstar,range(0,1,n)]
end
  
# Getting phi and phi*p using the slider value
df = @lift get_fig($vals)
phi = lift(v->v[1],df)
phiP = lift(v->v[2],df)
phiqstar = lift(v->v[3],df)
phiPqstar = lift(v->v[4],df)
Ran = lift(v->v[5],df)

# Initiate plot for phi
ax1 = Axis(figpphip[1,1],xlabel = "p", ylabel = "q")

# Plotting phi over p and q
GLMakie.contourf!(ax1,Ran,Ran,phi)
GLMakie.xlims!(0,1)
GLMakie.ylims!(0,1)
ax1.title = "phi"

# Initiate plot for phi*p
ax2 = Axis(figpphip[1,2],xlabel = "p", ylabel = "q")

# Plotting phi*p over p and q
GLMakie.contourf!(ax2,Ran,Ran,phiP)
GLMakie.xlims!(0,1)
GLMakie.ylims!(0,1)
ax2.title = "phi*p"

# Initiate plot for phi evaluated at q^*
ax3 = Axis(figpphip[2,1],xlabel = "p", ylabel = L"\phi|_{q^*}")

# Plotting phi|_q^* over p
GLMakie.plot!(ax3,Ran,phiqstar)
GLMakie.ylims!(60,250)

# Initiate plot for phi*p evaluated at q^*
ax4 = Axis(figpphip[2,2],xlabel = "p", ylabel = L"\phi|_{q^*}*p")

# Plotting p*phi|_q^* over p
GLMakie.plot!(ax4,Ran,phiPqstar)
GLMakie.ylims!(0,100)

# Activating widget window
GLMakie.activate!(title = "Invasion area viewer")

# Displaying widget
display(GLMakie.Screen(), figpphip)