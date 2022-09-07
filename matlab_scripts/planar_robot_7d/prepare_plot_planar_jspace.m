%% animation plot
%robot
f_anim = figure('Name','Animation','Position',[100 100 1200 500]);
ax_anim = subplot(1,2,1);
title('Planar robot')
xlabel('x, m')
ylabel('y, m')

axes(ax_anim)
ax_anim.XLim = [-12; 12];
ax_anim.YLim = [-12; 12];
hold on
%create robot
r_h = create_r(ax_anim,x_opt(:,1)',r,d,alpha,base);
plot(ax_anim, obs_pos(1),obs_pos(2),'r*')

[xc, yc] = circle_pts(obs_pos(1),obs_pos(2),obs_r-0.01);
crc_h = plot(ax_anim, xc, yc, 'r-','LineWidth',1.5);


% %joint-space
% x1_span=linspace(x1_min,x1_max,50); 
% x2_span=linspace(x2_min,x2_max,50); 
% 
% [X1_mg,X2_mg] = meshgrid(x1_span, x2_span);
% x=[X1_mg(:) X2_mg(:)]';
% x_dot = A*(x-attr);
% U_nominal = reshape(x_dot(1,:), length(x2_span), length(x1_span));
% V_nominal = reshape(x_dot(2,:), length(x2_span), length(x1_span));
% 
% %f_proj = figure(4);
% ax_proj= subplot(1,2,2);
% axes(ax_proj)
% axis equal
% title('C-space projection')
% xlabel('x1, rad')
% ylabel('x2, rad')
% 
% hold on
% ax_proj.XLim = [x1_min,x1_max];
% ax_proj.YLim = [x2_min,x2_max];
% 
% %nominal ds
% % obstacle
% inp = [x ; repmat(obs_pos,[1,length(x)])];
% val = y_f([inp; sin(inp); cos(inp)]);
% Z_mg = reshape(val,size(X1_mg));
% %distance heatmap
% [~, obs_h] = contourf(ax_proj, X1_mg,X2_mg,Z_mg,100,'LineStyle','none');
% %nominal ds
% l = streamslice(ax_proj, X1_mg,X2_mg,U_nominal,V_nominal);
% %obstacle shape
% [~, obs2_h] = contour(ax_proj, X1_mg,X2_mg,Z_mg,[obs_r,obs_r+thr],'LineStyle','-','LineColor','k','LineWidth',2);
% %attractor
% attr_h = plot(ax_proj, attr(1), attr(2), 'r*');
% fut_traj_h = plot(ax_proj, all_traj(:, 1, 1), all_traj(:, 2, 1),'g-o','LineWidth',2,'MarkerSize',3);
% past_traj_h = plot(ax_proj, x_opt(1, 1), x_opt(2, 1),'-ro','LineWidth',2,'MarkerSize',3);

%% functions
function pts = calc_fk(j_state,r,d,alpha,base)
    P = dh_fk(j_state,r,d,alpha,base);
    pts = zeros(3,3);
    for i = 1:1:length(j_state)+1
        v = [0,0,0];
        R = P{i}(1:3,1:3);
        T = P{i}(1:3,4);
        p = v*R'+T';
        pts(i,:) = p;
    end
end

function handle = create_r(ax_h,j_state,r,d,alpha,base)
    pts = calc_fk(j_state,r,d,alpha,base);
    handle = plot(ax_h,pts(:,1),pts(:,2),'LineWidth',2,...
        'Marker','o','MarkerFaceColor','k','MarkerSize',4);
end

function move_r(r_handle,j_state,r,d,alpha,base)
    pts = calc_fk(j_state,r,d,alpha,base);
    r_handle.XData = pts(:,1);
    r_handle.YData = pts(:,2);
end

function [xc,yc] = circle_pts(x,y,r)
    hold on
    th = linspace(0,2*pi,50);
    xc = r * cos(th) + x;
    yc = r * sin(th) + y;
end
