
%%Transcoder
function [mpeg_transcoded,difPframeMV,difPframeCoeff] = transmpeg(mpeg)
% Frame type
fpat = 'IPPPP'; % Custom

k = 0;
pf = [];
for i = 1:length(mpeg)-1

    [f, vectorX_matrix, vectorY_matrix] = decframe(mpeg{i},pf);
    fprintf('Frame %i decoded\n', i)
    k = k + 1;
    if k > length(fpat)
        k = 1;
    end
    ftype = fpat(k);

    [mpeg_transcoded{i},pf] = encframe(f,ftype,pf, vectorX_matrix, vectorY_matrix);
    % Cache previous frame
    sizeofmpeg=size(mpeg{i});
     totalmv(i)=calculatetotalmv(ftype,vectorX_matrix,vectorY_matrix,sizeofmpeg(1),sizeofmpeg(2));
      totalcoeff(i)=calculatetotalcoeff(mpeg{i});
      x1=totalmv(i)*0.5;
      x2=totalcoeff(i)*0.5;
      x3=x1+x2;
      threshold(i)=x1+x2;

     pf = f;
     %print current encoded frame
    fprintf('Frame %i encoded\n', i)
end
j=1;
uzun=size(totalmv)-1;
for i=1:uzun(2) 
    x=mpeg{i}(1,1);
    y=mpeg{i+1}(1,1);
    if(x.type == y.type)
         difPframeMV(j)=calculatedifferenceMV(totalmv(i),totalmv(i+1));
        difPframeCoeff(j)=calculatedifferenceCoef(totalcoeff(i),totalcoeff(i+1)); 
        difthres(j)=calculatedifferenceMV(threshold(i),threshold(i+1));
         j=j+1;
    end
end
filename = 'coef.xlsx';
T=table(difthres);
writetable(T,filename) 
end
%%
function [ fr, vectorX_matrix, vectorY_matrix ] = decframe( mpeg, pf )

mbsize = size(mpeg);%get the size of encoded frame
M = 16 * mbsize(1); %represents the number of pixels of a column
N = 16 * mbsize(2); %represetnts the number of pixels of a row

fr = zeros(M,N,3);
% Loop over macroblocks
for m = 1:mbsize(1)
    for n = 1:mbsize(2)
        % insert logo into lower right corner (m >= (mbsize(1) - 1) &&  n >=  (mbsize(2) - 1)
        % hight of logo = 2 macroblock, width of logo = 2 macroblock
        % rest of the blocks are decoded using video data
        x = 16*(m-1)+1 : 16*(m-1)+16; % calculate row numbers of current macroblock in frame
        y = 16*(n-1)+1 : 16*(n-1)+16;% calculate column numbers of current macroblock in frame
        %OUTPUTS of 'decmacroblock' FUNCTION
        %fr(x,y,:) : decoded macroblock
        
        %INPUTS of 'decmacroblock' FUNCTION
        %mpeg(m,n) : current encoded macroblock
        %pf : previous decoded frame
        %x : row numbers of current macroblock in frame
        %y : column numbers of current macroblock in frame
        
        %DESCRIPTION of 'decmacroblock' FUNCTION
        %decmacroblock : decodes macroblock without applying IDCT
        %rest of the blocks are decoded using current encoded frame 
        fr(x,y,:) = decmacroblock(mpeg(m,n),pf,x,y);
        vectorX_matrix(m,n) = mpeg(m,n).mvx;%store x component of the motion vectors of the current macroblock
        vectorY_matrix(m,n) = mpeg(m,n).mvy;%store y component of the motion vectors of the current macroblock
    
        
    end % macroblock loop
end % macroblock loop
end
%%
function mb = decmacroblock(mpeg,pf,x,y)

% Coeff quantization matrices
persistent q1 q2
if isempty(q1)
    q1 = qintra;
    q2 = qinter;
end

mb = zeros(16,16,3);%initialize macroblock

% Predict with motion vectors
if mpeg.type == 'P'
    mb = pf(x+mpeg.mvx,y+mpeg.mvy,:);
    q = q2;
else
    q = q1;
end

% Decode blocks
% IDCT is not computed
% Only quantization
for i = 6:-1:1
    coef = mpeg.coef(:,:,i) .* (mpeg.scale(i) * q) / 8;
    b(:,:,i) = idct2(coef);
end

% Construct macroblock
mb = mb + putblocks(b);
end
%%
function [mpeg,df] = encframe(f,ftype,pf, vectorX_matrix, vectorY_matrix)

[M,N,i] = size(f);
mbsize = [M, N] / 16;
mpeg = struct('type',[],'mvx',[],'mvy',[],'scale',[],'coef',[]);
mpeg(mbsize(1),mbsize(2)).type = [];

df = zeros(size(f));
for m = 1:mbsize(1)
    for n = 1:mbsize(2)
        % Encode one macroblock
        x = 16*(m-1)+1 : 16*(m-1)+16;% calculate row numbers of current macroblock in frame
        y = 16*(n-1)+1 : 16*(n-1)+16;% calculate column numbers of current macroblock in frame
        %OUTPUTS of 'encmacroblock' FUNCTION
        %mpeg(m,n) : current encoded macroblock
        %df(x,y,:) : previous decoded macroblock
        
        %INPUTS of 'encmacroblock' FUNCTION
        %f(x,y,:) : current decoded macroblock
        %pf : previous frame
        %x : row numbers of current macroblock in frame
        %y : column numbers of current macroblock in frame
        %vectorX_matrix(m,n) : x component of 
        %                                  current macroblock's motion vector
        %vectorY_matrix(m,n) : y component of
        %                                   current macroblock's motion vector 
        %DESCRIPTION of 'encmacroblock' FUNCTION
        %encmacroblock : encodes current macroblock using stored motion
        %vectors without applying DCT
        [mpeg(m,n),df(x,y,:)] = encmacroblock(f(x,y,:),ftype,pf,x,y,vectorX_matrix(m,n),vectorY_matrix(m,n));
        
    end % macroblock loop
end % macroblock loop
end
%%
function [mpeg,dmb] = encmacroblock(mb,ftype,pf,x,y,movx, movy)

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
    %OUTPUTS of 'getmotionvec' FUNCTION
    %mpeg : encoded macroblock
    %emb : error macroblock for motion compensation
    
    %INPUTS of 'getmotionvec' FUNCTION
    %mpeg :  current macroblock that will store motion vectors
    %mb : decoded macroblock
    %pf : previous frame
    %x : row numbers of current macroblock in frame
    %y : column numbers of current macroblock in frame
    %movx : x component of motion vector to be stored in encoded macroblock
    %movy : y component of motion vector to be stored in encoded macroblock
    
    %DESCRIPTION of 'getmotionvec' FUNCTION
    %asigns motion vectors to the macroblock that is being encoded
    %and calculates emb for motion compensation
    [mpeg,emb] = getmotionvec(mpeg,mb,pf,x,y,movx,movy);
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
%%
function [mpeg,emb] = getmotionvec(mpeg,mb,pf,x,y,m,n)
%set motion vectors
mpeg.mvx = m;
mpeg.mvy = n;
%calculate error macroblock for motion compensation
emb = mb - pf((x + m), y + n,:); % Error macroblock
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
function q = qinter
% Quantization table for P or B frames

% q = repmat(16,8,8);
q = 16;
end
%%
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

function total=calculatetotalmv(ftype,vectorX_matrix,vectorY_matrix,sizex,sizey)
total = 0;
if(ftype == 'P')
for k= 1:1:sizex
    for j = 1:1:sizey
        total = total+ abs(vectorX_matrix(k,j)) + abs(vectorY_matrix(k,j));
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
%%
function y=calculatetotalcoeff(mpeg)
y=0;
count=0;
c=0;
t=0;
kac=0;
sizeofmpeg=size(mpeg);
for k= 1:1:sizeofmpeg(1)
    for l = 1:1:sizeofmpeg(2)
        x=mpeg(k,l).coef;

        for i=1:8
            for j=1:8
                for z=1:6
                   kac=kac+1;
                    abc=x(i,j,z); %% deneme
                    if(abc==0)
                        count=count+1; %% kaç tane 0 var
                    else
                        c=c+abs(abc);
                    end
                     y=y+abs(abc);
                end
            end
        end
        
    end
end

end
