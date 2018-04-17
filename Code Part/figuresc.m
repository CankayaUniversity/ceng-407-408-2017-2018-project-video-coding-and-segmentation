function varargout = figure(input)


% Error checking
if any(sf < 0.0) | any(sf > 1)
    error('Ýnput girdisi 0-1 arasýnda olmalýdýr')
end

% Yükseklik ve geniþlik
if numel(input) == 1
    input = [input input];
end

% Pozisyonu hesapla
pos = [(1-input)/2, input];

% yazdýr
f = figure('Units','Normalized', ...
    'Position',pos(:)); % pos is always a vector

if nargout > 0
    varargout{1} = f;
end
