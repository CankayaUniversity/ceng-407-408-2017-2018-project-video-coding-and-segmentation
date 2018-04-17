function varargout = figure(input)


% Error checking
if any(sf < 0.0) | any(sf > 1)
    error('�nput girdisi 0-1 aras�nda olmal�d�r')
end

% Y�kseklik ve geni�lik
if numel(input) == 1
    input = [input input];
end

% Pozisyonu hesapla
pos = [(1-input)/2, input];

% yazd�r
f = figure('Units','Normalized', ...
    'Position',pos(:)); % pos is always a vector

if nargout > 0
    varargout{1} = f;
end
