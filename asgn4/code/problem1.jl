using Images, Optim, Grid, Interpolations
import PyPlot
include("./Common.jl");

function evaluate_flow(uv::Array{Float64, 3}, uv_gt::Array{Float64,3})
  u_diff = uv[:,:,1]-uv_gt[:,:,1]
  v_diff = uv[:,:,2]-uv_gt[:,:,2]
  ep = sqrt(u_diff.^2 + v_diff.^2)
  AEPE = sum(ep)/length(ep)

  return AEPE::Float64
end

function warp_image(im2::Array{Float64, 2}, uv0::Array{Float64, 3})
  im2_c = copy(im2)
  itp = interpolate(im2_c, BSpline(Linear()), OnCell())
  im_warp = zeros(Float64, size(im2))
  for j in 1:size(im2,2)
    for i in 1:size(im2,1)
      #if i+uv0[i,j,2] > 0 && j+uv0[i,j,1] > 0 && i+uv0[i,j,2] <= size(im2,1) && j+uv0[i,j,1] <= size(im2,2)
        im_warp[i,j] = itp[i+uv0[i,j,2],j+uv0[i,j,1]]
      #else
      #  im_warp[i,j] = 0.0
      #end
    end
  end
  return im_warp::Array{Float64, 2}
end

function load_images_and_flow()
    img1 = channelview(float64.(Gray.(load("../data/frame10.png"))))
    img2 = channelview(float64.(Gray.(load("../data/frame11.png"))))
    flow10 = read_flow_file("../data/flow10.flo")
    return img1::Array{Float64,2},img2::Array{Float64,2}, flow10::Array{Float64,3}
end

function compute_grad_images(im1::Array{Float64, 2}, im2::Array{Float64, 2}, uv0::Array{Float64,3})
  im2_w = warp_image(im2, uv0)
  It = im2_w - im1
  central_differences = 1/2*[-1 0 1]
  Ix = imfilter(im2_w, central_differences, "replicate")
  Iy = imfilter(im2_w, transpose(central_differences), "replicate")
  #Ix, Iy = imgradients(im2_w, Kernel.prewitt, "replicate")
  return Ix::Array{Float64, 2}, Iy::Array{Float64, 2}, It::Array{Float64, 2}
end

function logprior_HS(uv::Array{Float64,3}, sigma::Float64)
  N = size(uv,1)
  M = size(uv,2)
  res = zeros(Float64,(N,M))
  #TODO Boundary condition??
  for j in 1:M-1
    for i in 1:N-1
      res[i,j] = (uv[i,j,1] - uv[i+1,j,1])^2 + (uv[i,j,1] - uv[i,j+1,1])^2 + (uv[i,j,2] - uv[i+1,j,2])^2 + (uv[i,j,2] - uv[i,j+1,2])^2
    end
  end

  return -1/(2*sigma^2) * sum(res)
end

function test_grad_prior(uv::Array{Float64,3}, sigma::Float64)
  N = size(uv,1)
  M = size(uv,2)
  grad_u = zeros(Float64,(N,M))
  grad_v = zeros(Float64,(N,M))
  for j in 1:M
    for i in 1:N
      old_val = uv[i,j,1]
      r1 = logprior_HS(uv,sigma)
      uv[i,j,1] += 0.000001
      r2 = logprior_HS(uv,sigma)
      grad_u[i,j] = (r2-r1)/0.000001
      uv[i,j,1] = old_val
    end
  end

  for j in 1:M
    for i in 1:N
      old_val = uv[i,j,2]
      r1 = logprior_HS(uv,sigma)
      uv[i,j,2] += 0.000001
      r2 = logprior_HS(uv,sigma)
      grad_v[i,j] = (r2-r1)/0.000001
      uv[i,j,2] = old_val
    end
  end

  gu_computed, gv_computed = grad_logprior_HS(uv,sigma)
  display(gv_computed[2:end-1,2:end-1])
  display(grad_v[2:end-1,2:end-1])
  return sum(abs(gv_computed[2:end-1,2:end-1]-grad_v[2:end-1,2:end-1]))
end

function test_grad_likelihood(im1,im2,uv::Array{Float64,3}, uv0, sigma::Float64)
  N = size(uv,1)
  M = size(uv,2)
  grad_u = zeros(Float64,(N,M))
  grad_v = zeros(Float64,(N,M))
  Ix, Iy, It = compute_grad_images(im1, im2, uv0)
  for j in 1:M
    for i in 1:N
      old_val = uv[i,j,1]
      r1 = loglikelihood_HS(uv,uv0,Ix,Iy,It,sigma)
      uv[i,j,1] += 0.000001
      r2 = loglikelihood_HS(uv,uv0,Ix,Iy,It,sigma)
      grad_u[i,j] = (r2-r1)/0.000001
      uv[i,j,1] = old_val
    end
  end

  for j in 1:M
    for i in 1:N
      old_val = uv[i,j,2]
      r1 = loglikelihood_HS(uv,uv0,Ix,Iy,It,sigma)
      uv[i,j,2] += 0.000001
      r2 = loglikelihood_HS(uv,uv0,Ix,Iy,It,sigma)
      grad_v[i,j] = (r2-r1)/0.000001
      uv[i,j,2] = old_val
    end
  end

  gu_computed, gv_computed = grad_loglikelihood_HS(uv,uv0,Ix,Iy,It,sigma)
  display(gv_computed[2:end-1,2:end-1])
  display(grad_v[2:end-1,2:end-1])
  return sum(abs(gv_computed[2:end-1,2:end-1]-grad_v[2:end-1,2:end-1]))
end

function test_grad_posterior(im1,im2,uv::Array{Float64,3}, uv0, sigma::Float64)
  N = size(uv,1)
  M = size(uv,2)
  grad_u = zeros(Float64,(N,M))
  grad_v = zeros(Float64,(N,M))
  Ix, Iy, It = compute_grad_images(im1, im2, uv0)
  for j in 1:M
    for i in 1:N
      old_val = uv[i,j,1]
      r1 = logposterior_HS(uv,uv0,Ix,Iy,It,0.22,sigma)
      uv[i,j,1] += 0.000001
      r2 = logposterior_HS(uv,uv0,Ix,Iy,It,0.22,sigma)
      grad_u[i,j] = (r2-r1)/0.000001
      uv[i,j,1] = old_val
    end
  end

  for j in 1:M
    for i in 1:N
      old_val = uv[i,j,2]
      r1 = logposterior_HS(uv,uv0,Ix,Iy,It,0.22,sigma)
      uv[i,j,2] += 0.000001
      r2 = logposterior_HS(uv,uv0,Ix,Iy,It,0.22,sigma)
      grad_v[i,j] = (r2-r1)/0.000001
      uv[i,j,2] = old_val
    end
  end

  gu_computed, gv_computed = grad_logposterior_HS(uv,uv0,Ix,Iy,It,0.22,sigma)
  display(gu_computed[2:end-1,2:end-1])
  display(grad_u[2:end-1,2:end-1])
  return sum(abs(gu_computed[2:end-1,2:end-1]-grad_u[2:end-1,2:end-1]))
end

function grad_logprior_HS(uv::Array{Float64,3}, sigma::Float64)
  N = size(uv,1)
  M = size(uv,2)
  grad_u = zeros(Float64,(N,M))
  grad_v = zeros(Float64,(N,M))
  #TODO Boundary condition??
  for j in 2:M-1
    for i in 2:N-1
      grad_u[i,j] = 4*uv[i,j,1] - uv[i+1,j,1] - uv[i,j+1,1] - uv[i-1,j,1] - uv[i,j-1,1]
      grad_v[i,j] = 4*uv[i,j,2] - uv[i+1,j,2] - uv[i,j+1,2] - uv[i-1,j,2] - uv[i,j-1,2]
    end
  end
  grad_u *= -1/(sigma^2)
  grad_v *= -1/(sigma^2)

  return grad_u::Array{Float64,2}, grad_v::Array{Float64,2}
end

function loglikelihood_HS(uv::Array{Float64,3}, uv0::Array{Float64,3}, Ix::Array{Float64,2}, Iy::Array{Float64,2}, It::Array{Float64,2}, sigma::Float64)
  u_diff = uv[:,:,1] - uv0[:,:,1]
  v_diff = uv[:,:,2] - uv0[:,:,2]
  N = size(uv,1)
  M = size(uv,2)
  res = (Ix .* u_diff + Iy .* v_diff + It).^2
  return -1/(2*sigma^2) * sum(res)
end

function grad_loglikelihood_HS(uv::Array{Float64,3}, uv0::Array{Float64,3}, Ix::Array{Float64,2}, Iy::Array{Float64,2}, It::Array{Float64,2}, sigma::Float64)
  u_diff = uv[:,:,1] - uv0[:,:,1]
  v_diff = uv[:,:,2] - uv0[:,:,2]
  grad_u = -1/(sigma^2) * (Ix .* u_diff + Iy .* v_diff + It) .* Ix
  grad_v = -1/(sigma^2) * (Ix .* u_diff + Iy .* v_diff + It) .* Iy
  return grad_u::Array{Float64,2}, grad_v::Array{Float64,2}
end

function logposterior_HS(uv::Array{Float64,3}, uv0::Array{Float64,3}, Ix::Array{Float64,2}, Iy::Array{Float64,2}, It::Array{Float64,2}, lambda::Float64, sigma::Float64)
  lp = logprior_HS(uv, sigma)
  lh = loglikelihood_HS(uv, uv0, Ix, Iy, It, sigma)
  logposterior = lh + lambda*lp
  return logposterior::Float64
end

#function logposterior_HS_wrapper(im1::Array{Float64,2}, im2::Array{Float64,2}, uv::Array{Float64,1}, uv0::Array{Float64,3}, lambda::Float64, sigma::Float64)
function logposterior_HS_wrapper(uv::Array{Float64,1}, uv0::Array{Float64,3}, Ix, Iy, It, lambda::Float64, sigma::Float64)
  uv = reshape(uv, size(uv0))
  #Ix, Iy, It = compute_grad_images(im1, im2, uv)
  logposterior = -logposterior_HS(uv, uv0, Ix, Iy, It, lambda, sigma)
  return logposterior::Float64
end

function grad_logposterior_HS(uv::Array{Float64,3}, uv0::Array{Float64,3}, Ix::Array{Float64,2}, Iy::Array{Float64,2}, It::Array{Float64,2}, lambda::Float64, sigma::Float64)
  grad_lp_u, grad_lp_v = grad_logprior_HS(uv, sigma)
  grad_lh_u, grad_lh_v = grad_loglikelihood_HS(uv, uv0, Ix, Iy, It, sigma)
  grad_u = grad_lh_u + lambda*grad_lp_u
  grad_v = grad_lh_v + lambda*grad_lp_v
  return grad_u::Array{Float64,2}, grad_v::Array{Float64,2}
end

#function grad_logposterior_HS_wrapper(im1::Array{Float64,2}, im2::Array{Float64,2}, uv::Array{Float64,1}, uv0::Array{Float64,3}, lambda::Float64, sigma::Float64, storage::Array{Float64,1})
function grad_logposterior_HS_wrapper(uv::Array{Float64,1}, uv0::Array{Float64,3}, Ix, Iy, It, lambda::Float64, sigma::Float64, storage::Array{Float64,1})
  uv = reshape(uv, size(uv0))
  #Ix, Iy, It = compute_grad_images(im1, im2, uv)
  grad_u, grad_v = grad_logposterior_HS(uv, uv0, Ix, Iy, It, lambda, sigma)
  grad = zeros(Float64, size(uv0))
  grad[:,:,1] = -grad_u
  grad[:,:,2] = -grad_v
  storage[1:end] = grad[:]
  #uv0[1:end] = uv[1:end]
end

function flow_HS(im1::Array{Float64,2}, im2::Array{Float64,2}, uv0::Array{Float64,3}, lambda::Float64, sigma::Float64)
  Ix, Iy, It = compute_grad_images(im1, im2, uv0)
  f(uv) = logposterior_HS_wrapper(uv, uv0, Ix, Iy, It, lambda, sigma)
  #f(uv) = logposterior_HS_wrapper(im1, im2 ,uv, uv0, lambda, sigma)
  #g(uv,storage) = grad_logposterior_HS_wrapper(im1, im2, uv, uv0, lambda, sigma, storage)
  g(uv,storage) = grad_logposterior_HS_wrapper(uv, uv0, Ix, Iy, It, lambda, sigma, storage)
  res = optimize(f, g, zeros(size(uv0))[:], LBFGS(), Optim.Options(show_trace=true, iterations=200))
  uv_hs = reshape(Optim.minimizer(res), size(uv0))
  return uv_hs::Array{Float64,3}
end

function find_lambda(im1::Array{Float64,2},im2::Array{Float64,2},uv0::Array{Float64,3},uv_gt::Array{Float64,3},sigma::Float64)
  lambdas = collect(linspace(0.001,0.01,10))
  lambdaMin = lambdas[1]
  best_uv = flow_HS(im1, im2, uv0, lambdaMin, sigma)
  min_aepe = evaluate_flow(best_uv, uv_gt)
  for i in 2:length(lambdas)
    uv_i = flow_HS(im1, im2, uv0, lambdas[i], sigma)
    aepe_i = evaluate_flow(uv_i, uv_gt)
    if aepe_i < min_aepe
      min_aepe = aepe_i
      lambdaMin = lambdas[i]
      best_uv = uv_i
    end
  end
  return lambdaMin::Float64
end


function problem1()
  im1, im2, flow10 = load_images_and_flow()
  flow10[flow10 .> 1e9] = 0.0
  im2_w = warp_image(im2, flow10)
  #figure()
  #PyPlot.imshow(im1,"gray")
  #figure()
  #PyPlot.imshow(im2_w,"gray")
  figure()
  subplot(1,3,1)
  title("I1")
  PyPlot.imshow(im1, "gray")
  subplot(1,3,2)
  title("I2_w")
  PyPlot.imshow(im2_w, "gray")
  subplot(1,3,3)
  title("Difference between I2_w and I1")
  PyPlot.imshow(im2_w - im1, "gray")
  uv0 = zeros(Float64, size(flow10))
  #uv0 = rand(size(flow10))
  #uv0 = convert(Array{Float64,3},uv0)
  #uv0 = flow10
  #fill!(uv0, 1.0)
  lambda = 0.009
  #lambda = find_lambda(im1, im2, uv0, flow10, 1.0)
  #print("Best lambda was: $lambda\n")
  uv = zeros(Float64, size(flow10))
  for i in 1:2
    if i > 1
      uv0 = flow_HS(im1, im2, uv0, lambda, 1.0)
      uv += uv0
    else
      uv0 = flow_HS(im1, im2, uv0, lambda, 1.0)
      uv = uv0
    end
    im2 = warp_image(im2, uv0)
    uv0 = zeros(Float64, size(flow10))
    #figure()
    #PyPlot.imshow(copy(im2), "gray")
    #lambda = find_lambda(im1, im2, uv0, flow10, 1.0)
    #display(lambda)
  end
  uv_hs = uv
  aepe = evaluate_flow(uv_hs, flow10)
  print("Average end point error was $aepe\n")
  figure()
  PyPlot.imshow(flow_to_color(uv_hs))
  return uv_hs
end
