function boutput = reluBackward(binput, input)
    mask = max(input, 0);
    mask(mask>0) = 1;
    boutput = binput .* mask;
end
