function [boolian] = checkRC(Xtrain, Xcv)
    boolian = false;
    if (size(Xtrain, 1) == size(Xcv, 1))
        boolian = true;
    endif
end
