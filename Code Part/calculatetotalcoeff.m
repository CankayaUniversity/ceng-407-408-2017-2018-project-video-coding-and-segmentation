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