function [e_inf e_sup my] = enveloppe_inf_sup_signal(y, W)
% In the paper [1], this function allows to make the necessary processing
% to get the result in Section 2 (see figure 1)
% y is a 1D signal
% W is a window size, default=23
%
% [1] E. Ramasso, Investigating computational geometry for failure prognostics,
% International Journal on Prognostics and Health Management, 5(5):1-18, 2014.
%
% E. Ramasso, 2014
%

if nargin==1, W=23; end
s=smooth(y,W);
my=s;

%w=12; b=y; for i=w+1:length(b), b(i) = mean(y(i-w:i)); end
%figure,plot([y b])
%s=b; clear b

% s(1:4)=repmat(mean(s(1:4)),4,1);
% s(end-3:end)=repmat(mean(s(end-3:end)),4,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
supr=zeros(size(y));
supr(1:4)=max([s(1:4) ; y(1:4)]');
supr(end-3:end)=max([s(end-3:end) ; y(end-3:end)]');

f=find(y > s);
supr(f) = y(f);

suprfwd=supr;
for i=2:length(supr)
    if supr(i)==0
        suprfwd(i) = suprfwd(i-1);
    end
end
suprbwd=supr;
for i=length(supr)-1:-1:1
    if supr(i)==0
        suprbwd(i) = suprbwd(i+1);
    end
end
e_sup = (suprfwd+suprbwd)/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
infr=zeros(size(y));
infr(1:4)=min([s(1:4) ; y(1:4)]');
infr(end-3:end)=min([s(end-3:end) ; y(end-3:end)]');

f=find(y < s);
infr(f) = y(f);

infrfwd=infr;
for i=2:length(infr)
    if infr(i)==0
        infrfwd(i) = infrfwd(i-1);
    end
end
infrbwd=infr;
for i=length(infr)-1:-1:1
    if infr(i)==0
        infrbwd(i) = infrbwd(i+1);
    end
end
e_inf = (infrfwd+infrbwd)/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = find(e_inf > e_sup);
x = e_inf(f);
e_inf(f) = e_sup(f);
e_sup(f) = x;
f = find(e_inf == e_sup);
e_sup(f) = e_sup(f) + 0.1*e_sup(f);

f = find(e_inf > e_sup);    
    
if length(f)>0, 
    v=e_sup(f); 
    e_sup(f)=e_inf(f); 
    e_inf(f)=v; 
end

if 0
    figure,plot(s)
    hold on, plot(e_inf ,'g')
    hold on, plot(e_sup ,'r')
    
    sinfr2=smooth(e_inf ,15);
    ssupr2=smooth(e_sup ,15);
    figure,plot(s)
    hold on, plot(sinfr2,'g')
    hold on, plot(ssupr2,'r')
end

