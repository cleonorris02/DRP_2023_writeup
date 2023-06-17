using DifferentialEquations, LinearAlgebra, Plots, Flux, Statistics

NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
           Dense(1 => 3, tanh), 
           Dense(3 => 6, tanh),
           Dense(6 => 3))

NNODE(1.0)

ϵ = sqrt(eps(Float32))

# define Lorenz equations
function lorenz(u,p,t)
    [p[1]*(u[2]-u[1]); u[1]*(p[2]-u[3]) - u[2]; u[1]*u[2] - p[3]*u[3]]
   end
# define parameters
p = (10.0,28.0,8/3)

t=1.0

g(t) = t*NNODE(t) .+ [1f0,0f0,0f0] # initial condition

loss()=mean(norm((g(t+ϵ).-g(t))./ϵ .- lorenz(g(t),p,t),2) for t in 0:1f-2:0.5f0)

# train
opt = Flux.Descent(0.01) #0.01 is the starting pt of gradient descent
data = Iterators.repeated((), 10000) 
iter = 0
cb = function () # callback function to observe training
  global iter += 1
  if iter % 1000 == 0
    display(loss())
  end
end
display(loss())
@time Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)