%% actual ploting
past_traj_h.XData = x_opt(1, 1:mpciter+1);
past_traj_h.YData = x_opt(2, 1:mpciter+1);
fut_traj_h.XData = all_traj(:, 1, mpciter);
fut_traj_h.YData = all_traj(:, 2, mpciter);
axis equal
hold on
move_r(r_h,x_opt(:,mpciter)',r,d,alpha,base);
%pause(0.1)
drawnow


%% functions
function pts = calc_fk(j_state,r,d,alpha,base)
    P = dh_fk(j_state,r,d,alpha,base);
    pts = zeros(3,3);
    for i = 1:1:3
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
