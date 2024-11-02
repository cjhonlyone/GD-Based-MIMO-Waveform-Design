clear;
clc;

max_iterations=2000;

t = 0.01;

lset_opt(4,4,1,1,-20);

% M = [4,10];
% N = [32, 64, 128, 256, 512, 1024,2048];
% 
% for i = 1:length(M)
%     for j = 1:length(N)
%         tic;
% 
%         t_e = t/M(i)/N(j);
% 
%         w = ones(1,2*N(j)-1);
% 
%         [X, loss_history] = lset_opt(M(i),N(j),t_e,max_iterations);
%         X = X.';
%         tops = toc;
% 
%         CorrelationMatrix = GetCorrelationMatrix(X, M(i),N(j), 1);
%         PSLs = GetPSLOfCorrelationMatrix(CorrelationMatrix, M(i),N(j), 1);
%         ISLs = GetISLOfCorrelationMatrix(CorrelationMatrix, M(i),N(j), 1);
%         fprintf("lset_opt(M=%d, N=%d) PSL  = %2.2fdB, ISL  = %2.2fdB, tops = %2.2f s\n",M(i),N(j), max(PSLs), max(ISLs), tops);
% 
%         save(['lset_opt_t',num2str(t),'m',num2str(M(i)),'n',num2str(N(j)),'.mat'],"X", "PSLs", "ISLs", "tops", "loss_history");
%     end
% end

% M = [256];
% N = [1024];
% 
% for i = 1:length(M)
%     for j = 1:length(N)
%         tic;
% 
%         t_e = t/M(i)/N(j);
% 
%         w = ones(1,2*N(j)-1);
% 
%         [X, X_angle, loss_history] = lset_opt(M(i),N(j),t_e,max_iterations);
%         X = X.';
%         tops = toc;
% 
%         CorrelationMatrix = GetCorrelationMatrix(X, M(i),N(j), 1);
%         PSLs = GetPSLOfCorrelationMatrix(CorrelationMatrix, M(i),N(j), 1);
%         ISLs = GetISLOfCorrelationMatrix(CorrelationMatrix, M(i),N(j), 1);
%         fprintf("lset_opt(M=%d, N=%d) PSL = %2.2f dB, ISL = %2.2f dB, tops = %2.2f s\n",M(i),N(j), max(PSLs), max(ISLs), tops);
% 
%         save(['lset_opt_t',num2str(t),'m',num2str(M(i)),'n',num2str(N(j)),'.mat'],"X", "PSLs", "ISLs", "tops", "loss_history");
%     end
% end

% M = [4];
% N = [1024];
% 
% for i = 1:length(M)
%     for j = 1:length(N)
%         tic;
% 
%         t_e = t/M(i)/N(j);
% 
%         w = ones(1,2*N(j)-1);
% 
%         [X, X_angle, loss_history] = lset_opt_bi(M(i),N(j),t_e,max_iterations);
%         X = X.';
%         tops = toc;
% 
%         CorrelationMatrix = GetCorrelationMatrix(X, M(i),N(j), 1);
%         PSLs = GetPSLOfCorrelationMatrix(CorrelationMatrix, M(i),N(j), 1);
%         ISLs = GetISLOfCorrelationMatrix(CorrelationMatrix, M(i),N(j), 1);
%         fprintf("lset_opt_bi(M=%d, N=%d) PSL = %2.2f dB, ISL = %2.2f dB, tops = %2.2f s\n",M(i),N(j), max(PSLs), max(ISLs), tops);
% 
%         % save(['lset_opt_t',num2str(t),'m',num2str(M(i)),'n',num2str(N(j)),'.mat'],"X", "PSLs", "ISLs", "tops", "loss_history");
%     end
% end

% CorrelationMatrix = abs(x').^2/N/N;
% CorrelationMatrix(N,([1:M]-1)*M+[1:M]) = ones(1,M);
% 

% for ii = N
%     load(['lset_opt_t',num2str(t),'m4n',num2str(ii),'.mat'])
% 
%     yrad = X;
%     [N, M] = size(yrad);
%     G = 0;
%     E = N-1;
%     CorrelationMatrix = GetCorrelationMatrix(yrad, M, N, 1);
%     PSLs = GetPSLOfCorrelationMatrix(CorrelationMatrix, M, N, 1);
%     ISLs = GetISLOfCorrelationMatrix(CorrelationMatrix, M, N, 1);
%     fprintf("%s\t\t PSL  = %2.2fdB, ISL  = %2.2fdB, tops = %2.2f s\n", num2str(ii), max(PSLs), max(ISLs), tops);
% end

% yrad = X;
% [N, M] = size(yrad);
% G = 0;
% E = N-1;
% CorrelationMatrix = GetCorrelationMatrix(yrad, M, N, 1);
% 
% SubCorrelationMatrix = ExtractSubCorrelationMatrix(CorrelationMatrix, M, N, 1, 1, 0);
% if M~= 1
%     GenerateFigureOfCorrelation(CorrelationMatrix, M, N, 1, 'Auto', 'on', 'FixedYlim', ['M ', num2str(M), ' N ', num2str(N), ' Auto-Correlation WPSL'])
%     GenerateFigureOfCorrelation(CorrelationMatrix, M, N, 1, 'Cross', 'on', 'FixedYlim', ['M ', num2str(M), ' N ', num2str(N), ' Cross-Correlation WPSL'])
% else
%     GenerateFigureOfCorrelation(CorrelationMatrix, M, N, 1, 'Auto', 'on', 'FixedYlim', ['M ', num2str(M), ' N ', num2str(N), ' Auto-Correlation WPSL'])
% end
% 
% figure;
% plot(loss_history)
% 
% 


