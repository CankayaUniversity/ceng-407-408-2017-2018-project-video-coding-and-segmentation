function shots=encodemov(str)
vid = VideoReader(str)
[difmv,difcoef,difthres]=encodemovie(vid)
j=1;
for i=1:size(difthres)
    if(difthres(i) >threshold)
    shots(j)=difthres(i);
    j=j+1;
    end
end
formatSpec = '%s%s%s';
A3='.xlsx';
str=sprintf(formatSpec,A2,'-mv',A3);
filename = str;
str=sprintf(formatSpec,A2,'-coef',A3);
filename2 = str;
T=table(difmv);
writetable(T,filename)
Y=table(difcoef);
writetable(Y,filename2);
str=sprintf(formatSpec,A2,'-thres',A3);
filename3 = str;
Z=table(difthres);
writetable(Z,filename3)
end
function [difPframeMV,difPframeCoeff,difthres] = encodemovie(vid)

framepatern = 'IPPPP';
pf = [];
t=1;
w1=0,5;
w2=0,5;
 while hasFrame(vid)
 mov = readFrame(vid);
k=mod(t,5);
if(k==0)
    k=5;
end
for i = 1:size(mov,4)
    
    fr = double(mov(:,:,:,i));
    fr = rgb2ycc(fr);
        ftype = framepatern(k);
    
    [mpeg{t},pf] = encodeframe(fr,ftype,pf);
    fprintf('%d frame is encoded\n',t);
    totaldc(t)=calculatetotalcoeff(mpeg{t});
    totalmv(t)=calculatemv(mpeg{t});
    x1(t)=totalmv(t)*w1;
    x2(t)=totaldc(t)*w2;
    threshold(t)=x1(t)+x2(t);

end
t=t+1;

 end
j=1;
uzun=size(totalmv)-1;
for i=1:uzun(2) 
    x=mpeg{i}(1,1);
    y=mpeg{i+1}(1,1);
    if(x.type == y.type)
         difPframeMV(j)=calculatedifferenceMV(totalmv(i),totalmv(i+1));
        difPframeCoeff(j)=calculatedifferenceCoef(totaldc(i),totaldc(i+1)); 
        difthres(j)=calculatedifferenceMV(threshold(i),threshold(i+1));
         j=j+1;
    end
end
end
function [encmpeg,df] = encodeframe(frame,ftype,pf)

[M,N,i] = size(frame);
mbsize = [M, N] / 16;
encmpeg = struct('type',[],'mvx',[],'mvy',[],'scale',[],'coef',[]);
encmpeg(mbsize(1),mbsize(2)).type = [];

% Loop over macroblocks
pfy = pf(:,:,1);
df = zeros(size(frame));
for m = 1:mbsize(1)
    for n = 1:mbsize(2)
        
        % Encode one macroblock
        x = 16*(m-1)+1 : 16*(m-1)+16;
        y = 16*(n-1)+1 : 16*(n-1)+16;
        [encmpeg(m,n),df(x,y,:)] = encmacroblock(frame(x,y,:),ftype,pf,pfy,x,y);
        
    end % macroblock loop
end % macroblock loop

end
function [encmpeg,dmb] = encmacroblock(mb,ftype,pf,pfy,x,y)

% Coeff quantization matrices
persistent q1 q2
if isempty(q1)
    q1 = qintra;
    q2 = qinter;
end

% Quality scaling
scale = 31;

% Init mpeg struct
encmpeg.type = 'I';
encmpeg.mvx = 0;
encmpeg.mvy = 0;

% Find motion vectors
if ftype == 'P'
    encmpeg.type = 'P';
    [encmpeg,emb] = getmv(encmpeg,mb,pf,pfy,x,y);
    mb = emb; % Set macroblock to error for encoding
    q = q2;
else
    q = q1;
end

% Get lum and chrom blocks
b = getblocks(mb);

% Encode blocks
for i = 6:-1:1
    encmpeg.scale(i) = scale;
    coef = dct2(b(:,:,i));
    encmpeg.coef(:,:,i) = round( 8 * coef ./ (scale * q) );
end

% Decode this macroblock for reference by a future P frame
dmb = decmacroblock(encmpeg,pf,x,y);

end
function [encmpeg,emb] = getmv(encmpeg,mb,pf,pfy,x,y)

% Do search in Y only
mby = mb(:,:,1);
[M,N] = size(pfy);

step = 8;

dx = [0 1 1 0 -1 -1 -1  0  1]; 
dy = [0 0 1 1  1  0 -1 -1 -1];
                               

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

encmpeg.mvx = mvx; % Store motion vectors
encmpeg.mvy = mvy;

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

persistent k
if isempty(k)
    k = dctmtx(8);
end

y = k * x * k';

end
%%
function y = idct2(x)

persistent k
if isempty(k)
    k = dctmtx(8);
end

y = k' * x * k;

end
function q = qinter

q = 16;

end
function mb = decmacroblock(mpeg,pf,x,y)

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
function y=calculatetotalcoeff(mpeg)
y=0;
count=0;
sizeofmpeg=size(mpeg);
for k= 1:1:sizeofmpeg(1)
    for l = 1:1:sizeofmpeg(2)
        x=mpeg(k,l).coef;

        for i=1:8
            for j=1:8
                for z=1:6
                    abc=x(i,j,z); %% deneme
                    if(abc==0)
                        count=count+1; %% kaç tane 0 var
                    end
                     y=y+abs(abc);
                end
            end
        end
        
    end
end

end
function y=calculatemv(mpeg)
y=0;
if(mpeg(1,1).type=='I')
    y=0;
else
    sizeofmpeg=size(mpeg);
for k= 1:1:sizeofmpeg(1)
    for l = 1:1:sizeofmpeg(2)
        mvx=mpeg(k,l).mvx;
        mvy=mpeg(k,l).mvy;
        y=y+abs(mvx)+abs(mvy);
    end
end
end

end
function difference=calculatedifferenceMV(x,y)
difference=abs(abs(y)-abs(x));
difference=(difference*100)/abs(x);
end
function difference=calculatedifferenceCoef(x,y)
difference=abs(abs(y)-abs(x));
difference=(difference*100)/abs(x);
end
