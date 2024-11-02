
function [X, X_angle, loss_history] = lset_opt(M,N,t,max_iterations, Pobj)
reverseStr  = '';
% adam
alpha = 0.01; % learning rate
beta1 = 0.9; % decay rate for first moment estimate
beta2 = 0.999; % decay rate for second moment estimate
epsilon = 1e-8; % constant to avoid division by zero

a_angle = randn(M,N);
a_complex = exp(1j*a_angle);
da_complexE = complex(zeros(size(a_angle)));

x = complex(zeros(M*M,2*N-1));
ga1 = complex(zeros(M*M,4*N-3));
ga2 = complex(zeros(M*M,4*N-3));

% Initialize variables
m = zeros(size(a_angle)); % first moment estimate
vv = zeros(size(a_angle)); % second moment estimate

% Initialize loss history vector
loss_history = zeros(max_iterations, 1);

% load myrad1.mat
% a_angle = reshape(brad, [N,M]).';
% a_angle_sym = sym('a_angle_sym',[M,N]);
% da_angle_sym = sym('da_angle_sym',[M,N]);

% indiceu = zeros(M*(M+1)/2, 3);
% 
% t = 1;
% for u = 1:M
%     for v = u:M
%         indiceu(t,1) = (u-1)*M+v; 
%         indiceu(t,2) = u; 
%         indiceu(t,3) = v; 
%         t = t + 1;
%     end
% end


for epoch = 1:1:max_iterations

    a_complex = exp(1j*a_angle);
    % a_angle

    % tic
    % for u = 1:M
    %     for v = 1:M
    %         x((u-1)*M+v,:) = xcorr(a_complex(v,:), a_complex(u,:));
    %     end
    % end
    % x(indiceu(:,1),:) = xcorr(a_complex(indiceu(:,3),:), a_complex(indiceu(:,2),:));
    % toc;

    % tic
    x = GetCorrelationMatrixxx(a_complex.', M, N, 1).';
    % toc

    % x
    % cmdouble = reshape(abs(x).^2,[M*M,2*N-1]).'
    % x
    % abs(x).^2.'

    x(([1:M]-1)*M+[1:M],N) = x(([1:M]-1)*M+[1:M],N) .* zeros(M,1);
    % x = x./N;

    xm = abs(x).^2 /N/N;

    g1 = x.*exp((xm - lset(xm, t))/t )/N/N;

    for u = 1:M
        for v = 1:M
            ga1((u-1)*M+v,:) = xcorr([a_complex(v,:), zeros(1, N-1)], g1((u-1)*M+v,:));
        end
    end

    for u = 1:M
        da_complexE(u,:) = sum(ga1((u-1)*M+[1:M],(N):(2*N-1)),1);
    end
    % ga1.'
    % ga2.'

    da_angleE = 2*imag(conj(a_complex).*da_complexE);
    % da_angleE

    % Update first moment estimate
    m = beta1 * m + (1 - beta1) * da_angleE;
    % Update second moment estimate
    vv = beta2 * vv + (1 - beta2) * (da_angleE.^2);
    % Correct bias in first and second moment estimates
    m_hat = m / (1 - beta1^epoch);
    v_hat = vv / (1 - beta2^epoch);

    learning_rate = alpha ./ (sqrt(v_hat) + epsilon);

    a_angle = a_angle - learning_rate .* m_hat;

    % fprintf("epoch %d, ISL %.2f, PSL %.2f\b", t, 10*log10(ISL/N/N), 10*log10(PSL/N/N));

    ISL = sum(xm, [1,2]);
    PSL = max(max(xm));



%     msg                 = sprintf('lset_opt(M=%d, N=%d) PSL = %.2f dB, ISL = %.2f dB, epoch %d',M,N, 10*log10(PSL), 10*log10(ISL),epoch);
%     fprintf([reverseStr, msg]);
%     reverseStr = repmat(sprintf('\b'), 1, length(msg));

    loss_history(epoch) = 10*log10(PSL);

    % if epoch < 2500
    % elseif epoch < 6000
    %     alpha = alpha * 0.9995;
    % elseif epoch < 8000
    %     alpha = alpha * 0.9998;
    % elseif epoch < 10000
    %     alpha = alpha * 0.9999;
    % else
    %     alpha = alpha * 0.99999;
    % end

    if (10*log10(PSL) < Pobj)
        break;
    end
end
X_angle = a_angle;
X = exp(1j*a_angle);
% fprintf("\n");
end