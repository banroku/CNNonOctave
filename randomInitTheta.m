function [theta] = randomInitTheta(theta)
    
    epsilon = sqrt(6)/sqrt(sum(size(theta)));
    theta = rand(size(theta)) * 2 * epsilon - epsilon;
    
end
