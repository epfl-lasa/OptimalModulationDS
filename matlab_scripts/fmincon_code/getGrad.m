function [val, grad] = getGrad(dlnet,x_batch)
    n_batch = size(x_batch,2);
    dim = size(x_batch,1);
    val = zeros(1,n_batch);
    grad = zeros(dim,n_batch);
    parfor i = 1:n_batch
        [val(i),grad(:,i)] = dlfeval(@compute_gradient,dlnet,x_batch(:,i));
    end
end

function [val,gradients]=compute_gradient(dlnet,dlx)
    val = forward(dlnet,dlx);
    gradients = dlgradient(val,dlx);%automatic gradient
end
