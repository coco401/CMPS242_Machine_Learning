% Keep digit precision
format long g

% Get train set and put into matrix
filepath = 'C:\Users\GWCP\Documents\CMPS242_Machine_Learning\hw1\train.txt'
train_set = csvread(filepath)

% Get x and t Nx1 matrices out of train_set
x = train_set(:,1)
t = train_set(:,2)

% Randomize indexes
rand_idx = randperm(100)
rand_idx = rand_idx(:)

% Randomize Nx1 matrices
x = x(rand_idx)
t = t(rand_idx)

% Tranpose x matrix
x = transpose(x)

% Create 10xN matrix
% Multiplying each element in each row to the power of N from 0-9
X = [ x.^0; x.^1; x.^2; x.^3; x.^4; x.^5; x.^6; x.^7; x.^8; x.^9 ]

% Create lambda array values
% exp(-4) equal to 1 * e(^4)
lambda = [0, exp(-4), exp(-3), exp(-2), exp(-1), 1, 2, 3,4,5, 10, 20, 50, 100];

% 100-fold Leave-one-out algorithm
for fold=1:100
    if fold == 1
        X_train = X(:, [2:100]);
        T_train = t([2:100],:);
        X_valid = X(:, [1:1]);
        T_valid = t([1:1],:);
    elseif fold == 100
        X_train = X(:, [1:99]); 
        T_train = t([1:99], :);
        X_valid = X(:, [99:100]);
        T_valid = t([99:100], :);
    else
        X_train = X(:, [1:(fold-1) fold+1:100]);
        T_train = t([1:(fold-1) fold+1:100], :);
        X_valid = X(:, [fold:fold]);
        T_valid = t([fold:fold], :);
    end
    
    for i=1:length(lambda)
        w_star = inv(X_train * transpose(X_train) + lambda(i) * eye(10)) * X_train * T_train;
        error(fold, i) = (0.5) * sum((transpose(X_valid) * w_star - T_valid).^2);
    end
end

error_mean = mean(error);

% Find min error and index to obtain lambda value
[M,I] = min(error_mean);

% Get test set
filepath = 'C:\Users\GWCP\Documents\CMPS242_Machine_Learning\hw1\test.txt'
test_set = csvread(filepath)

% Get and parse x_test and t_test Nx1 matrices out of test_set
x_test = test_set(:,1)
t_test = test_set(:,2)

% Transpose x_test
x_test = transpose(x_test);

% Create X_test and T_test matrices
X_test = [ x_test.^0; x_test.^1; x_test.^2; x_test.^3; x_test.^4; x_test.^5; x_test.^6; x_test.^7; x_test.^8; x_test.^9 ];
T_test = t_test;

% Find min w_star value
final_w_star = inv(X * transpose(X) + lambda(I) * eye(10)) * X * t;
error_test = (0.5) *  sum((transpose(X_valid) * final_w_star - T_valid).^2);

% Get Error Root Square Mean
for k=1:length(lambda)

    % E_rms for test set
    final_w_star_plot_test = inv(X * transpose(X) + lambda(k) * eye(10)) * X * t;
    error_test(k) = (0.5) * sum((transpose(X_test) * final_w_star_plot_test - T_test).^2);
    e_rms_test(k) = sqrt(2*error_test(k)/ 5000);
    
    % E_rms for train set
    final_w_star_plot_train = inv(X * transpose(X) + lambda(k) * eye(10)) * X * t;
    error_train(k) = (0.5) * sum((transpose(X) * final_w_star_plot_train - t).^2);
    e_rms_train(k) = sqrt(2*error_train(k)/ 100);
end

% Converting lambda values into logarithmic function for plotting
for p=1:length(lambda)
    test_lm(p) = log(lambda(p));
end

% Plot E_RMS and ln(lambda) for training and test set
%figure
%plot(test_lm, e_rms_train, '-o')
%hold on
%plot(test_lm, e_rms_test, '-o')
%title('Regularizaton: E_{RMS} vs ln(\lambda)')
%xlabel('ln(\lambda)')
%ylabel('E_{RMS}')
%legend('Training','Test')
%hold off

% Getting best w_star value with min lambda
coef = final_w_star;

% Plot w* training with train data points
f = @(x) coef(10)*x^9 + coef(9)*x^8 + coef(8)*x^7 + coef(7)*x^6 + coef(6)*x^5 + coef(5)*x^4 + coef(4)*x^3 + coef(3)*x^2 + coef(2)*x^1 +  coef(1);

hold on
plot(x_test,t_test,'oy')
plot(x,t,'ob')
fplot(f, 'k')
title('Leave One Out Method')
xlabel('x')
ylabel('t')
legend('Test Data Points','Train Data Points', 'Fitting Cruve')
hold off
