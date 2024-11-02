function lse = lset(x,t)
max_x = max(x(:));

lse = t*log(sum(exp((x(:)-max_x)/t))) + max_x;

end
