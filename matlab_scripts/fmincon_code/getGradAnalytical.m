function [val, grad] = getGradAnalytical(y_f, dy_f,x_batch)
    n_batch = size(x_batch,2);
    dim = size(x_batch,1);
    %val = zeros(1,n_batch);
    grad = zeros(dim,n_batch);
    val = y_f(x_batch);
    for i = 1:n_batch
        grad(:,i) = dy_f(x_batch(:,i));
    end
end
