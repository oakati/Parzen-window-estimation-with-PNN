function [test_targets] = Nearest_Neighbor(train_patterns, train_targets, test_patterns, Knn) %param_struct_knn

% Classify using the Nearest neighbor algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%	Knn		        - Number of nearest neighbors 
%
% Outputs
%	test_targets	- Predicted targets

L			= length(train_targets);
Uc          = unique(train_targets);

if (L < Knn)
   error('You specified more neighbors than there are points.')
end

N               = size(test_patterns, 2);
test_targets    = zeros(1,N); 
for i = 1:N
    dist            = sum((train_patterns - test_patterns(:,i)*ones(1,L)).^2);
    [m, indices]    = sort(dist);
    
    n(:,i)               = hist(train_targets(indices(1:Knn)), Uc);
    
    [m, best(i)]       = max(n(:,i));
    
    test_targets(i) = Uc(best(i));
end

n_ = n./sum(n,1);

% Uclasses = unique(test_targets);
% 
% for i = Uclasses
%     indices = (test_targets == i);
%     b = n_(:,indices);
%     a = max(b);
%     param_struct_knn(i).p_error_given_x = (sum(b)-a)./sum(b);
% end
