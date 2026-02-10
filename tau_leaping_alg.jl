using PoissonRandom

function tau_leaping_alg(S::Int64,W::Int64,N::Int64,q::Float64,p::Float64,C::Int64,tau::Float64,tauBeta::Float64,k::Float64)
    while S < W && S + W < C # Runs whilst max size hasnt been met and W>S
        pks = p*k*S
        qw = q*W
        W += pois_rand(tauBeta*(W-qw+pks)) # Updates W using the total rate of W births
        S += pois_rand(tauBeta*(qw+k*S-pks)) # Updates S using the total rate of S births
        N += 1 # Increments timestep
    end
    return S, N*tau
end