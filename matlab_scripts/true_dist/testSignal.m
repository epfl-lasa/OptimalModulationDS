clc
clear all
close all
rps = 0.1288;
goal_rpd = 500;

allo = 100000:1e4:1e6;
signal = 1000:10:50000;
for i = 1:1:length(signal)
    for j = 1:1:length(allo)
        a = allo(j);
        s = signal(i);
        r_all = s*rps;
        rpm_cur(i,j) = r_all/a*1e6;
        rpm_add(i,j) = r_all/(a+1e6)*1e6;
        if rpm_add(i,j)<300 && rpm_cur(i,j)>500 
            msg = sprintf(['Signal: %4.2f, Allo: %4.2f, Rew_in: ' ...
                '%4.2f, Rew_new: %4.2f'], s, a, rpm_cur(i,j), rpm_add(i,j));
            disp(msg)
        end
    end
end
%% heatmaps
% [allo_mg,signal_mg] = meshgrid(allo, signal);
% figure(1)
% contourf(allo_mg,signal_mg,rpm_cur,100,'LineStyle','none');
% xlabel('allo')
% ylabel('signal')
% title('current')
% figure(2)
% contourf(allo_mg,signal_mg,rpm_add,100,'LineStyle','none');
% xlabel('allo')
% ylabel('signal')
% title('added')
% hold on
% contour(allo_mg,signal_mg,rpm_add,[199,200],'LineWidth',2,'Color','k');
% figure(3)
% contourf(allo_mg,signal_mg,rpm_cur./rpm_add,100,'LineStyle','none');
% xlabel('allo')
% ylabel('signal')
% title('ratio')
%% surfs
[allo_mg,signal_mg] = meshgrid(allo, signal);
figure(1)
hold on
surf(allo_mg,signal_mg,rpm_cur,'EdgeColor','none','FaceColor',[0.4660 0.6740 0.1880]);
surf(allo_mg,signal_mg,rpm_add,'EdgeColor','none','FaceColor',[0 0.4470 0.7410]);
surf(allo_mg,signal_mg,rpm_cur*0+200,'EdgeColor','none','FaceColor',[0.8500 0.3250 0.0980]);
xlabel('allocation')
ylabel('signal')
zlabel('reward')
zlim([0 2000])
% figure(2)
% contourf(allo_mg,signal_mg,rpm_add,100,'LineStyle','none');
% xlabel('allo')
% ylabel('signal')
% title('added')
% hold on
% contour(allo_mg,signal_mg,rpm_add,[199,200],'LineWidth',2,'Color','k');
% figure(3)
% contourf(allo_mg,signal_mg,rpm_cur./rpm_add,100,'LineStyle','none');
% xlabel('allo')
% ylabel('signal')
% title('ratio')
% 
