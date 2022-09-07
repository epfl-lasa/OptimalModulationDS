function [y_f, tmp] = tanhNN(net)
    w1 = double(net.Layers(2).Weights);
    b1 = double(net.Layers(2).Bias);
    w2 = double(net.Layers(4).Weights);
    b2 = double(net.Layers(4).Bias);
    w3 = double(net.Layers(6).Weights);
    b3 = double(net.Layers(6).Bias);
    w4 = double(net.Layers(8).Weights);
    b4 = double(net.Layers(8).Bias);
    w5 = double(net.Layers(10).Weights);
    b5 = double(net.Layers(10).Bias);

    z1 = @(x) w1*x+b1;
    h1 = @(x)tanh(z1(x));
    z2 = @(x) w2*h1(x)+b2;
    h2 = @(x)tanh(z2(x));
    z3 = @(x) w3*h2(x)+b3;
    h3 = @(x)tanh(z3(x));
    z4 = @(x) w4*h3(x)+b4;
    h4 = @(x)tanh(z4(x));
    z5 = @(x) w5*h4(x)+b5;

    %y_f is a forward prop for the layers above
    y_f = @(x)z5(x);
    tmp = 0;
%     dy_f = @(x) w4*diag(1-tanh(z3(x)).^2)*...
%             w3*diag(1-tanh(z2(x)).^2)*...
%             w2*diag(1-tanh(z1(x)).^2)*w1;
end