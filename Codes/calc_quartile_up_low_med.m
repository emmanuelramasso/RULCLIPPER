function [med lowerquartile upperquartile] = calc_quartile_up_low_med(tmp)
% Compute the median, lower and upper quartile

if length(tmp)==0
    med=nan;
    lowerquartile=nan;
    upperquartile=nan;
    return
end

if length(tmp)==1
    med=tmp;
    lowerquartile=tmp;
    upperquartile=tmp;
    return
end


med=median(tmp);
tmp = sort(tmp,'ascend');
if ndims(tmp)==2 && (size(tmp,1)==1 || size(tmp,2)==1), tmp=tmp(:); end

if mod(size(tmp,1),2)==0
    n=size(tmp,1)/2;
    tmp1 = tmp(1:n,:);
    tmp2 = tmp(n+1:end,:);
else
    n=floor(size(tmp,1)/2);
    tmp1 = tmp(1:n,:);
    tmp2 = tmp(n+1:end,:);
end

% on prend le milieu du sup
n2=size(tmp2,1);
if mod(n2,2)==0
    u1 = tmp2(n2/2,:);
    u2 = tmp2(n2/2+1,:);
    upperquartile = (u1+u2)/2;
else
    upperquartile = tmp2(ceil(n2/2),:);    
end

% on prend le milieu du inf
n1=size(tmp1,1);
if mod(n1,2)==0
    u1 = tmp1(n1/2,:);
    u2 = tmp1(n1/2+1,:);
    lowerquartile = (u1+u2)/2;
else
    lowerquartile = tmp1(ceil(n1/2),:);    
end
