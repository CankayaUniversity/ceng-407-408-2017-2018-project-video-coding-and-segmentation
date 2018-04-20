function mvplot
% P frame üzerindeki mvleri gösteren fonksiyon
load lastmov

[X,Y] = size(mpeg{1});

for f = 1:length(mpeg)
    if mpeg{f}(1,1).type == 'I'
        continue
    end
    for i = 1:X
        for j = 1:Y
            mvx(i,j) = mpeg{f}(i,j).mvy;
            mvy(i,j) = mpeg{f}(i,j).mvx;
        end
    end
    figuresc(0.8)
    quiver(flipud(mvx),flipud(mvy))
    set(gca,'XLim',[-1,Y+2],'YLim',[-1, X+2])
    title(sprintf('Motion vectors for frame %i',f))
end