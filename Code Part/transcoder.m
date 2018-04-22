
function transcoder
fprintf('\nContent Based Video Segmentation\n')
%number of frames, 0 to transcode all frames
nf = 0;
fprintf('Transcoding all frames\n')
%load video encoded with mpegproj.m
load lastmov
%load logo encoded with encodeLogo.m
load logo
%start timer for transcode time
tic
mov3 = transmpeg(mpeg,encodedLogo);
fprintf('Transcode time: %s\n', sec2timestr(toc))
%save transcoded video(mov3) with logo insertion
save transcoder mov3

%%Transcoder
function mpeg_transcoded = transmpeg(mpeg,logo)

% Frame type
fpat = 'IPPPP'; % Custom

% Loop over frames
k = 0;
pf = [];
progressbar
for i = 1:length(mpeg)
    %OUTPUTS of 'decframe' FUNCTION
    %f : decoded frame
    %vectorX_matrix : x components of motion vectores
    %vectorY_matrix : y components of motion vectors
    
    %INPUTS of 'decframe' FUNCTION
    %mpeg{i} : current encoded frame
    %pf : previous frame
    %logo : encoded logo
    
    %DESCRIPTION of 'decframe' FUNCTION
    %decframe : decodes frame, inserts logo and
    %stores motion vectors to use later in encframe function
    [f, vectorX_matrix, vectorY_matrix] = decframe(mpeg{i},pf,logo);
    %print current decoded frame number 
    fprintf('Frame %i decoded\n', i)
    % Get frame type
    k = k + 1;
    if k > length(fpat)
        k = 1;
    end
    ftype = fpat(k);

    % Encode frame
    %OUTPUTS of 'encframe' FUNCTION
    %mpeg_transcoded{i} : encoded frame
    %pf : previous frame
    
    %INPUTS of 'encframe' FUNCTION
    %f :current decoded frame
    %ftype : frame type I or P
    %pf : previous decoded frame
    %vectorX_matrix : x components of motion vectores
    %vectorY_matrix : y components of motion vectors
    
    %DESCRIPTION of 'encframe' FUNCTION
    %encode frame : encodes the decoded current frame and instead of
    %recalculating motion information uses stored motion information by
    %decframe
    [mpeg_transcoded{i},pf] = encframe(f,ftype,pf, vectorX_matrix, vectorY_matrix);
    % Cache previous frame
     totalmv(i)=calculatetotalmv(ftype,vectorX_matrix,vectorY_matrix);
      totalcoeff(i)=calculatetotalcoeff(mpeg{i});
     pf = f;
     %print current encoded frame
    fprintf('Frame %i encoded\n', i)
    %update progressbar
    progressbar( i / length(mpeg))
   
    
end
j=1;
for i=1:250 
    x=mpeg{i}(1,1);
    y=mpeg{i+1}(1,1);
    if(x.type == y.type)
         difPframeMV(j)=calculatedifferenceMV(totalmv(i),totalmv(i+1));
         difPframeCoeff(j)=calculatedifferenceCoef(totalcoeff(i),totalcoeff(i+1));
           if(difPframeMV(j) >= 41)
               for k=1:i
               mov5(k)=mpeg(k);
               end
       %    save last2mov mov5
        %   playlast(5);


           end
               
         j=j+1;
    end
end

fprintf('hello');

%%
function [ fr, vectorX_matrix, vectorY_matrix ] = decframe( mpeg, pf, logo )

%Calculate the size of frame in pixel
mbsize = size(mpeg);%get the size of encoded frame
M = 16 * mbsize(1); %represents the number of pixels of a column
N = 16 * mbsize(2); %represetnts the number of pixels of a row
counter = 1;%initialize counter for logo macroblocks to be decoded

fr = zeros(M,N,3);%initialize frame
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
        if (m >= (mbsize(1) - 1) &&  n >=  (mbsize(2) - 1)  )%insert logo to the 4 macroblocks located on the lower right corner of the frame
           if (counter == 1)
               fr(x,y,:) = decmacroblock(logo(1,1),pf,1:16,1:16);%first logo macroblock to be inserted
           elseif (counter == 2)
               fr(x,y,:) = decmacroblock(logo(1,2),pf,17:32, 1:16);%second logo macroblock to be inserted
           elseif (counter == 3)
               fr(x,y,:) = decmacroblock(logo(2,1),pf,1:16, 17:32);%third logo macroblock to be inserted
           elseif(counter == 4)
               fr(x,y,:) = decmacroblock(logo(2,2),pf,17:32, 17:32);%forth logo macroblock to be inserted
           end
           counter = counter + 1;
        else
        % motion vectors of blocks around the logo is set to 0 (including logo)
        % to decrease the rate of drift problem
        % motion vectors of logo inserted macroblocks are also set to 0 due
        % to logo is in a fixed position
            if(m >= (mbsize(1) - 2) &&  n >=  (mbsize(2) - 2)  ) 
                mpeg(m,n).mvx = 0;
                mpeg(m,n).mvy = 0;
            end
        %rest of the blocks are decoded using current encoded frame 
        fr(x,y,:) = decmacroblock(mpeg(m,n),pf,x,y);
        vectorX_matrix(m,n) = mpeg(m,n).mvx;%store x component of the motion vectors of the current macroblock
        vectorY_matrix(m,n) = mpeg(m,n).mvy;%store y component of the motion vectors of the current macroblock
        end
        
    end % macroblock loop
end % macroblock loop

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

%%
function [mpeg,emb] = getmotionvec(mpeg,mb,pf,x,y,m,n)
%set motion vectors
mpeg.mvx = m;
mpeg.mvy = n;
%calculate error macroblock for motion compensation
emb = mb - pf((x + m), y + n,:); % Error macroblock

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

%%
function q = qinter
% Quantization table for P or B frames

% q = repmat(16,8,8);
q = 16;

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

function total=calculatetotalmv(ftype,vectorX_matrix,vectorY_matrix)
total = 0;
if(ftype == 'P')
for k= 1:1:18
    for j = 1:1:22
        total = total+ abs(vectorX_matrix(k,j)) + abs(vectorY_matrix(k,j));
    end
end

end

function difference=calculatedifferenceMV(x,y)
difference=abs(abs(y)-abs(x));
difference=(difference*100)/abs(x);

function difference=calculatedifferenceCoef(x,y)
difference=abs(abs(y)-abs(x));
difference=(difference*100)/abs(x);

%%
function y=calculatetotalcoeff(mpeg)
y=0;
count=0;
t=0;
for k= 1:1:18
    for l = 1:1:22
        x=mpeg(k,l).coef;
        for i=1:8
            for j=1:8
                for z=1:6
                    abc=x(i,j,z); %% deneme
                    if(x(i,j,z)==0)
                        count=count+1; %% kaç tane 0 var
                    end
                    % y=y+abs(abc);
                   t=t+abc;
                end
            end
        end
        
        y=sum(sum(sum(mpeg(k,l).coef)))+y;
        y=y+(count*65); %% 0 yerine 1 yaz,255 yaz,65 yaz
    end
end

