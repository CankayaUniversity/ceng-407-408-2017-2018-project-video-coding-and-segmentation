[filename pathname ] = uigetfile({'*.*'},'Choose a video')
formatSpec = '%s%s';
A1=pathname
A2=filename
str = sprintf(formatSpec,A1,A2)
vid = VideoReader(str)
duration=vid.Duration;
name=vid.Name;
frame=vid.FrameRate;
mpeg=encmov(vid)
[mov3,difmv,difcoef] = transmpeg(mpeg);
implay(str)
formatSpec = '%s%s%s';
A3='.xlsx';
str=sprintf(formatSpec,A2,'-mv',A3)
filename = str;
str=sprintf(formatSpec,A2,'-coef',A3)
filename2 = str;
T=table(difmv);
writetable(T,filename)
Y=table(difcoef);
writetable(Y,filename2);
save transcoder mpeg
function mpeg = encmov(vid)

% Frame type pattern (repeats for entire movie)
fpat = 'IPPPP'; % Custom
% fpat = 'I'; % All I frames
% fpat = ['I', repmat('P',1,size(mov,4))]; % All P frames (except #1)
pf = [];
t=1;
 while hasFrame(vid)
 mov = readFrame(vid);
k=mod(t,5);
if(k==0)
    k=5;
end
% Loop over frames
for i = 1:size(mov,4)
    
    % Get frame
    f = double(mov(:,:,:,i));
    % Convert frame to YCrCb
    f = rgb2ycc(f);
    
    % Get frame type
    ftype = fpat(k);
    
    % Encode frame
    [mpeg{t},pf] = encframe(f,ftype,pf);
    
    
end
t=t+1;
end
save mov
end
function [mpeg,df] = encframe(f,ftype,pf)

[M,N,i] = size(f);
mbsize = [M, N] / 16;
mpeg = struct('type',[],'mvx',[],'mvy',[],'scale',[],'coef',[]);
mpeg(mbsize(1),mbsize(2)).type = [];

% Loop over macroblocks
pfy = pf(:,:,1);
df = zeros(size(f));
for m = 1:mbsize(1)
    for n = 1:mbsize(2)
        
        % Encode one macroblock
        x = 16*(m-1)+1 : 16*(m-1)+16;
        y = 16*(n-1)+1 : 16*(n-1)+16;
        [mpeg(m,n),df(x,y,:)] = encmacroblock(f(x,y,:),ftype,pf,pfy,x,y);
        
    end % macroblock loop
end % macroblock loop

end
function [mpeg,dmb] = encmacroblock(mb,ftype,pf,pfy,x,y)

% Coeff quantization matrices
persistent q1 q2
if isempty(q1)
    q1 = qintra;
    q2 = qinter;
end

% Quality scaling
scale = 31;

% Init mpeg struct
mpeg.type = 'I';
mpeg.mvx = 0;
mpeg.mvy = 0;

% Find motion vectors
if ftype == 'P'
    mpeg.type = 'P';
    [mpeg,emb] = getmotionvec(mpeg,mb,pf,pfy,x,y);
    mb = emb; % Set macroblock to error for encoding
    q = q2;
else
    q = q1;
end

% Get lum and chrom blocks
b = getblocks(mb);

% Encode blocks
for i = 6:-1:1
    mpeg.scale(i) = scale;
    coef = dct2(b(:,:,i));
    mpeg.coef(:,:,i) = round( 8 * coef ./ (scale * q) );
end

% Decode this macroblock for reference by a future P frame
dmb = decmacroblock(mpeg,pf,x,y);

end
function [mpeg,emb] = getmotionvec(mpeg,mb,pf,pfy,x,y)

% Do search in Y only
mby = mb(:,:,1);
[M,N] = size(pfy);

% % Exhaustive search
% maxstep = 10; % Largest allowable motion vector in x and y
% 
% mvxv = -maxstep:maxstep;
% mvyv = -maxstep:maxstep;
% minsad = inf;
% for i = 1:length(mvxv)
%     
%     tx = x + mvxv(i);
%     if (tx(1) < 1) | (M < tx(end))
%         continue
%     end
%     
%     for j = 1:length(mvyv)
%         
%         ty = y + mvyv(j);
%         if (ty(1) < 1) | (N < ty(end))
%             continue
%         end
%         
%         sad = sum(sum(abs(mby-pfy(tx,ty))));
%         
%         if sad < minsad
%             minsad = sad;
%             mvx = mvxv(i);
%             mvy = mvyv(j);
%         end
%         
%     end
% end

% Logarithmic search
step = 8; % Initial step size for logarithmic search

dx = [0 1 1 0 -1 -1 -1  0  1]; % Unit direction vectors
dy = [0 0 1 1  1  0 -1 -1 -1]; % [origin, right, right-up, up, left-up,
                               %         left, left-down, down, right-down]

mvx = 0;
mvy = 0;
while step >= 1
    
    minsad = inf;
    for i = 1:length(dx)
        
        tx = x + mvx + dx(i)*step;
        if (tx(1) < 1) || (M < tx(end))
            continue
        end
        
        ty = y + mvy + dy(i)*step;
        if (ty(1) < 1) || (N < ty(end))
            continue
        end
        
        sad = sum(sum(abs(mby-pfy(tx,ty))));
        
        if sad < minsad
            ii = i;
            minsad = sad;
        end
        
    end
    
    mvx = mvx + dx(ii)*step;
    mvy = mvy + dy(ii)*step;
    
    step = step / 2;
    
end

mpeg.mvx = mvx; % Store motion vectors
mpeg.mvy = mvy;

emb = mb - pf(x+mvx,y+mvy,:); % Error macroblock

end
function q = qintra
% Quantization table for I frames

q = [ 8 16 19 22 26 27 29 34;
     16 16 22 24 27 29 34 37;
     19 22 26 27 29 34 34 38;
     22 22 26 27 29 34 37 40;
     22 26 27 29 32 35 40 48;
     26 27 29 32 35 40 48 58;
     26 27 29 34 38 46 56 69;
     27 29 35 38 46 56 69 83 ];

end
%%
function y = dct2(x)
% Perform 2-D DCT

% Use dctmtx to compute IDCT faster
persistent d
if isempty(d)
    d = dctmtx(8);
end

y = d * x * d';

% % DCT is seperable so compute on columns, then on rows
% y = dct(x); % Columns
% y = dct(y')'; % Rows
end
%%
function y = idct2(x)
% Perform 2-D IDCT

% Use dctmtx to compute IDCT faster
persistent d
if isempty(d)
    d = dctmtx(8);
end

y = d' * x * d;

% % DCT is seperable so compute on columns, then on rows
% y = idct(x); % Columns
% y = idct(y')'; % Rows
end
function q = qinter
% Quantization table for P or B frames

% q = repmat(16,8,8);
q = 16;

end
function mb = decmacroblock(mpeg,pf,x,y)

% Coeff quantization matrices
persistent q1 q2
if isempty(q1)
    q1 = qintra;
    q2 = qinter;
end

mb = zeros(16,16,3);

% Predict with motion vectors
if mpeg.type == 'P'
    mb = pf(x+mpeg.mvx,y+mpeg.mvy,:);
    q = q2;
else
    q = q1;
end

% Decode blocks
for i = 6:-1:1
    coef = mpeg.coef(:,:,i) .* (mpeg.scale(i) * q) / 8;
    b(:,:,i) = idct2(coef);
end

% Construct macroblock
mb = mb + putblocks(b);
end
function mb = putblocks(b)

mb = zeros([16, 16, 3]);

% Four lum blocks
mb( 1:8,  1:8,  1) = b(:,:,1);
mb( 1:8,  9:16, 1) = b(:,:,2);
mb( 9:16, 1:8,  1) = b(:,:,3);
mb( 9:16, 9:16, 1) = b(:,:,4);

% Two subsampled chrom blocks
z = [1 1; 1 1];
mb(:,:,2) = kron(b(:,:,5),z);
mb(:,:,3) = kron(b(:,:,6),z);

end
%%
function b = getblocks(mb)

b = zeros([8, 8, 6]);

% Four lum blocks
b(:,:,1) = mb( 1:8,  1:8,  1);
b(:,:,2) = mb( 1:8,  9:16, 1);
b(:,:,3) = mb( 9:16, 1:8,  1);
b(:,:,4) = mb( 9:16, 9:16, 1);

% Two subsampled chrom blocks (mean of four neighbors)
b(:,:,5) = 0.25 * ( mb(1:2:15,1:2:15, 2) + mb(1:2:15,2:2:16, 2) ...
                  + mb(2:2:16,1:2:15, 2) + mb(2:2:16,2:2:16, 2) );
b(:,:,6) = 0.25 * ( mb(1:2:15,1:2:15, 3) + mb(1:2:15,2:2:16, 3) ...
                  + mb(2:2:16,1:2:15, 3) + mb(2:2:16,2:2:16, 3) );
end

%%
function ycc = rgb2ycc(rgb)

% Transformation matrix
m = [ 0.299     0.587     0.144;
     -0.168736 -0.331264  0.5;
      0.5      -0.418688 -0.081312];

% Get movie data
[nr,nc,c] = size(rgb);

% Reshape for matrix multiply
rgb = reshape(rgb,nr*nc,3);

% Transform color coding
ycc = m * rgb';
ycc = ycc + repmat([0; 0.5; 0.5],1,nr*nc);

% Reshape to original size
ycc = reshape(ycc',nr,nc,3);
end
