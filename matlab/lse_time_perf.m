clear;
clc;
test_vector = zeros(15,4);
test_vector(:,1:3) = [[4,64,-17];
[4,64,-18];
[4,64,-19];
[4,256,-23];
[4,256,-24];
[4,256,-25];
[4,1024,-27];
[4,1024,-28];
[4,1024,-29];
[4,1024,-30];
[10,1024,-25];
[10,1024,-26];
[10,1024,-27];
[10,1024,-28];
[128,1024,-25]];

reverseStr  = '';
%% Optimization
repeat_t = 10;
for i = 1:1
    M = test_vector(i, 1);
    N = test_vector(i, 2);
    Pobj = test_vector(i, 3);
    fprintf(['Test ',num2str(i),' M ',num2str(M),' N ',num2str(N),' Pobj ',num2str(Pobj),'\n'])
    for r = 1:repeat_t

        fprintf(['Round ',num2str(r),'... \n'])

        t = 0.01;

        t_e = t/M/N;

        w = ones(1,2*N-1);

        tic
        [X, X_angle, loss_history] = lset_opt(M,N,0.00000954,20000,Pobj);
        % X = X.';
        tops = toc;
        test_vector(i,4) = test_vector(i,4) + tops;
        
        fprintf(['    tops ',num2str(test_vector(i,4)),' inner ',num2str(r),'\n'])
    end
end
test_vector(:,4) = test_vector(:,4)/repeat_t;



