using Images
using PyPlot

function read_flow_file(filename)

TAG_FLOAT = 202021.25;  # check for this when READING the file

# sanity check
if isempty(filename) == 1
    error("readFlowFile: empty filename");
    return 0;
end

idx = rsearch(filename, ".");
idx = idx[end];

if length(filename[idx:end]) == 1
    println(sprintf("readFlowFile: extension required in filename", filename));
    return 0;
end

if (filename[idx:end] == ".flo") != 1
    println(sprintf("readFlowFile: filename %s should have extension .flo", filename));
    return 0;
end



fid = open(filename, "r");
isopen(fid)
if (isopen(fid) != true)
    println(sprintf("readFlowFile: could not open: ",filename));
    return 0;
end

tag     = read(fid, Float32);
width   = read(fid, Int32);
height  = read(fid, Int32);

# sanity check

if (tag != TAG_FLOAT)
    println(sprintf("readFlowFile(%s): wrong tag (possibly due to big-endian machine?)",filename))
    return 0;
end

if (width < 1 || width > 99999)
    println(sprintf("readFlowFile(%s): illegal width %i",filename,width));
    return 0;
end

if (height < 1 || height > 99999)
    println(sprintf("readFlowFile(%s): illegal height %i",filename,height));
    return 0;
end

nBands = 2;

# arrange into matrix form
tmp = read(fid, Float32, height * width* nBands);
tmp = convert(Array{Float64}, tmp);
tmp = reshape(tmp, (Int64(width*nBands), Int64(height)));
tmp = tmp';
img = similar(tmp,Int64(height),Int64(width),Int64(nBands));
img[:,:,1] = tmp[:, (1:width)*nBands-1];
img[:,:,2] = tmp[:, (1:width)*nBands];

close(fid)
return img;

end

function flow_to_color(flow)

  UNKNOWN_FLOW_THRESH = 1e9;
  UNKNOWN_FLOW = 1e10;

  height, widht, nBands = size(flow);

  if nBands != 2
      println("flow_to_color: image must have two bands");
  end

  u = flow[:,:,1];
  v = flow[:,:,2];

  maxu = -999;
  maxv = -999;

  minu = 999;
  minv = 999;
  maxrad = -1;

  # fix unknown flow
  idxUnknown = (abs(u) .> UNKNOWN_FLOW_THRESH) | (abs(v) .> UNKNOWN_FLOW_THRESH) ;
  u[idxUnknown] = 0;
  v[idxUnknown] = 0;

  maxu = max(maxu, maximum(u));
  minu = min(minu, minimum(u));

  maxv = max(maxv, maximum(u));
  minv = min(minv, minimum(v));

  rad = sqrt(u.^2+v.^2);
  maxrad = max(maxrad, maximum(rad));

  # println(maxrad, " ", minu, " ", maxu, " ", minv, " ", maxv);

  u = u./(maxrad+eps());
  v = v./(maxrad+eps());

  # compute color
  img = compute_color(u, v);
  # unknown flow
  IDXm = zeros(Bool, size(idxUnknown,1), size(idxUnknown,2), 3);
  IDXm[:,:,1] = idxUnknown;
  IDXm[:,:,2] = idxUnknown;
  IDXm[:,:,3] = idxUnknown;
  img[IDXm] = 0;
  img = convert(Array{UInt8,3},img);
  return img;
end

function compute_color(u,v)

  nanIdx = isnan(u) | isnan(v);
  u[nanIdx] = 0;
  v[nanIdx] = 0;

  rows, cols = size(u);
  colormap = zeros(rows, cols, 3);

  colorwheel = make_colorwheel();
  ncols = size(colorwheel, 1);

  rad = sqrt(u.^2+v.^2);

  a = atan2(-v, -u)/pi;

  fk = (a+1) /2 * (ncols-1) + 1;  # -1~1 maped to 1~ncols

  k0 = floor(Int64, fk);                 # 1, 2, ..., ncols

  k1 = k0+1;
  k1[k1.==ncols+1] = 1;

  f = fk - k0;

  for i = 1:size(colorwheel,2)
      tmp = colorwheel[:,i];
      col0 = tmp[k0]/255;
      col1 = tmp[k1]/255;
      col = (1-f).*col0 + f.*col1;

      idx = rad .<= 1;
      col[idx] = 1-rad[idx].*(1-col[idx]);    # increase saturation with radius

      col[~idx] = col[~idx]*0.75;             # out of range

      #colormap[:,:,i] = convert(Array{UInt8,2},floor(255*col.*(1-nanIdx)));
      colormap[:,:,i] = floor(255*col.*(1-nanIdx));
  end

  return colormap

end

function make_colorwheel()
  RY = 15;
  YG = 6;
  GC = 4;
  CB = 11;
  BM = 13;
  MR = 6;
  ncols = RY + YG + GC + CB + BM + MR;
  colorwheel = zeros(ncols, 3);

  col = 0;
  #RY
  colorwheel[1:RY, 1] = 255;
  colorwheel[1:RY, 2] = floor(255*collect(0:RY-1)/RY);
  col = col+RY;

  #YG
  colorwheel[col+collect(1:YG), 1] = 255 - floor(255*collect(0:YG-1)/YG);
  colorwheel[col+collect(1:YG), 2] = 255;
  col = col+YG;

  #GC
  colorwheel[col+collect(1:GC), 2] = 255;
  colorwheel[col+collect(1:GC), 3] = floor(255*collect(0:GC-1)/GC);
  col = col+GC;

  #CB
  colorwheel[col+collect(1:CB), 2] = 255 - floor(255*collect(0:CB-1)/CB);
  colorwheel[col+collect(1:CB), 3] = 255;
  col = col+CB;

  #BM
  colorwheel[col+collect(1:BM), 3] = 255;
  colorwheel[col+collect(1:BM), 1] = floor(255*collect(0:BM-1)/BM);
  col = col+BM;

  #MR
  colorwheel[col+collect(1:MR), 3] = 255 - floor(255*collect(0:MR-1)/MR);
  colorwheel[col+collect(1:MR), 1] = 255;

  return colorwheel
end
