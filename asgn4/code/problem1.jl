using Images, Optim, Grid, Interpolations
import PyPlot
include("./Common.jl");

function evaluate_flow(uv::Array{Float64, 3}, uv_gt::Array{Float64,3})
  u_diff = uv[:,:,1]-uv_gt[:,:,1]
  v_diff = uv[:,:,2]-uv_gt[:,:,2]
  ep = sqrt(u_diff.^2 + v_diff.^2)
  ep_filt = [x for x in ep if x < 1000000000]
  AEPE = sum(ep_filt)/length(ep)

  return AEPE::Float64
end

function warp_image(im2::Array{Float64, 2}, uv0::Array{Float64, 3})
  itp = interpolate(im2, BSpline(Linear()), OnGrid())
  im_warp = zeros(Float64, size(im2))
  for j in 1:size(im2,2)
    for i in 1:size(im2,1)
      if uv0[i,j,2] < 1000000000 || uv0[i,j,1] < 1000000000
        im_warp[i,j] = itp[i+uv0[i,j,2],j+uv0[i,j,1]]
      else
        im_warp[i,j] = im2[i,j]
      end
    end
  end
  return im_warp::Array{Float64, 2}
end

function load_images_and_flow()
    img1 = 255.*channelview(float64.(Gray.(load("../data/frame10.png"))))
    img2 = 255.*channelview(float64.(Gray.(load("../data/frame11.png"))))
    flow10 = read_flow_file("../data/flow10.flo")
    return img1::Array{Float64,2},img2::Array{Float64,2}, flow10::Array{Float64,3}
end

function compute_grad_images(im1::Array{Float64, 2}, im2::Array{Float64, 2}, uv0::Array{Float64,3})
  im2_w = warp_image(im2, flow10)
  It = im2_w - im1
  Iy, Ix = imgradients(im2_w, Kernel.prewitt, "replicate")
  return Ix::Array{Float64, 2}, Iy::Array{Float64, 2}, It::Array{Float64, 2}
end

function logposterior_HS(uv::Array{Float64,3}, uv0::Array{Float64,3}, Ix::Array{Float64,2}, Iy::Array{Float64,2}, It::Array{Float64,2}, lambda::Float64, sigma::Float64)

  return logposterior::Float64
end

function grad_logposterior_HS(uv::Array{Float64,3}, uv0::Array{Float64,3}, Ix::Array{Float64,2}, Iy::Array{Float64,2}, It::Array{Float64,2}, lambda::Float64, sigma::Float64)

  return grad_u::Array{Float64,2}, grad_v::Array{Float64,2}
end

function flow_HS(im1::Array{Float64,2}, im2::Array{Float64,2}, uv0::Array{Float64,3}, lambda::Float64, sigma::Float64)

  return uv_hs::Array{Float64,3}
end

function find_lambda(im1::Array{Float64,2},im2::Array{Float64,2},uv0::Array{Float64,3},uv_gt::Array{Float64,3},sigma::Float64)

  return lambdaMin::Float64
end


function problem1()
  im1, im2, flow10 = load_images_and_flow()
  im2_w = warp_image(im2, flow10)
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
end
