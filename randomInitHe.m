function [theta] = randomInitHe(theta, iw, ic)
    
    sigma = sqrt( 2 / (iw^2 * ic) );
    theta = randn(size(theta)) * sigma;
    
end
