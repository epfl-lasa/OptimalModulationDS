clc
clear all

l_f = @(x, l1, l2, d1, d2, k)l1 + (l2 - l1)./(1 + exp(-k*(x - (d1+d2)/2)));

x = -5:0.01:15;
d1 = 3;
d2 = 10;
k = 3;
l_n = l_f(x, 0, 1, d1, d2, k);
l_t = l_f(x, 2, 1, d1, d2, k);

f = figure(1);
ax_h = axes(f);
hold on
axis equal
set(gcf,'color','w');
xlabel('x_1')
ylabel('x_2')
set(gca,'fontsize',14)
ylim([-1,3])
xlim([x(1), x(end)])
plot(x, l_n, 'LineWidth',2)
plot(x, l_t, 'LineWidth',2)