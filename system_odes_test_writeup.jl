using DifferentialEquations
using Plots

# define Lorenz equations
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
   end

# initial condition
u0 = [1.0,0.0,0.0]

# timespan to solve on
tspan = (0.0,100.0)

# define parameters
p = (10.0,28.0,8/3)

# solve Lorenz system
using DifferentialEquations
prob = ODEProblem(lorenz,u0,tspan,p)
sol = solve(prob)
plot(sol,vars=(1,2,3)) #butterfly image