function CM = GetCorrelationMatrixxx(WaveformMatrix, M, N, Q)
% GetCorrelationMatrix.
%
%   F = GetCorrelationMatrix(WaveformMatrix, M, N, Q) returns a  
%   Quantized CorrelationMatrix.
%
%   GetCorrelationMatrix properties:
%       WaveformMatrix    - N rows M columns wavefore matrix
%       M                 - Number of channels
%       N                 - Sequence length
%       Q                 - Quantized coefficient
% 
%   Example:
%
    
%   Copyright 2023 Caojiahui

% if Q~=1
%     q = quantizer([Q Q],'saturate');
%     WaveformMatrix = quantize(q,WaveformMatrix);
% end

L = 2*N;

CM_s = fft(WaveformMatrix, L);
CM_re1 = repmat(CM_s, 1, M);
CM_re2 = reshape(repmat(CM_s, M, 1), [L, M*M]);

CM = complex(zeros(L, M*M));
% tri_indices = tril(ones(M))==1;
tri_indices = ones(M)==1;
CM(:,tri_indices) = fftshift((ifft(CM_re1(:,tri_indices).*conj(CM_re2(:,tri_indices)))), 1);
% CM(:,tri_indices) = CM(:,tri_indices).^2;
CM = CM(1+(L-2*N+1):end, :);

end