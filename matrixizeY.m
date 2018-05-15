function Ymatrix = matrixizeY(Y,K);
    Ymatrix = zeros(K, size(Y,2));
    for i = 1:K
        Ymatrix(i,:) = (Y==i);
    end
end

