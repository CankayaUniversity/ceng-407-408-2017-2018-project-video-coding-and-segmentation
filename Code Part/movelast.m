function movelast(k)


if isempty(k)
   
k = 5;

end


load transcoder

%loading .mat

for i = 1:size(mov3,4)
 
%starting to end
   c{i}.cdata = uint8(mov3(:,:,:,i));
  %%4-d dimensional video
 c{i}.colormap = [];

//set color
end


figuresc([0.9 0.5])
movie(c,k,10)
