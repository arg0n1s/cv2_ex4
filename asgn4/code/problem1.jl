using Images, Optim, Grid
import PyPlot
include("./Common.jl");

function evaluate_flow(uv::Array{Float64, 3}, uv_gt::Array{Float64,3})

  return AEPE::Float64
end

function warp_image(im2::Array{Float64, 2}, uv0::Array{Float64, 3})

  return im_warp::Array{Float64, 2}
end

function compute_grad_images(im1::Array{Float64, 2}, im2::Array{Float64, 2}, uv0::Array{Float64,3})

  return Ix::Array{Float64, 2}, Iy::Array{Float64, 2}, It::Array{Float64, 2}
end

function logposterior_HS(uv::Array{Float64,3}, uv0::Array{Float64,3}, Ix::Array{Float64,2}, Iy::Array{Float64,2}, It::Array{Float64,2}, lambda::Float64, sigma::Float64)

  return logposterior::Float64
end

function grad_logposterior_HS(uv::Array{Float64,3}, uv0::Array{Float64,3}, Ix::Array{Float64,2}, Iy::Array{Float64,2}, It::Array{Float64,2}, lambda::Float64, sigma::Float64)

  return grad_u::Array{Float64,2}, grad_v::Array{Float64,2}
end

function flow_HS(im1::Array{Float64,2}, im2::Array{Float64,2}, uv0::Array{Float64,3}, lambda::Float64, sigma::Float64)

  return uv_hs::::Array{Float64,3}
end

function find_lambda(im1::Array{Float64,2},im2::Array{Float64,2},uv0::Array{Float64,3},uv_gt::Array{Float64,3},sigma::Float64)

  return lambdaMin::Float64
end
