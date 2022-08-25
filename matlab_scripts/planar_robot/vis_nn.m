clc
close all force
clear all
global joint_state

%gmm model creation 
load('data/net50_pos_thr.mat')
[y_f, ~] = tanhNN(net);

%parameters definition
r = [4 4 0];
d = [0 0 0];
alpha = [0 0 0];
base = eye(4);
k1 = 0.95; k2 = 0.95;
q_min = [-k1*pi, -k2*pi, -10,-10];
q_max = [k1*pi, k2*pi, 10, 10];
q_init = [ -1.48,1.14,0.85,-1.3];
joint_state = q_init;

%figure with robot
fig_handle = figure('Name','Two-dimensional robot');
ax_h = axes(fig_handle,'View',[0 90]);
axis equal
hold on
ax_h.XLim = [-2.2 2.2];
ax_h.YLim = [-2.2 2.2];
ax_h.ZLim = [-0.1 0.1];

%create contour object
x_span = linspace(q_min(3),q_max(3));
y_span = linspace(q_min(4),q_max(4));
[X_mg,Y_mg] = meshgrid(x_span, y_span);
[~, ctr] = contourf(ax_h, X_mg,Y_mg,0*X_mg,100,'LineStyle','none');
lvl = 0.5;
[~, cr1] = contour(ax_h, X_mg,Y_mg,X_mg-10,[lvl,lvl+0.001],'LineStyle','-','LineColor','k','LineWidth',2);

%create robot and dot objects
r_h = create_r(ax_h,joint_state(1:2),r,d,alpha,base);
dot = plot(ax_h, joint_state(3),joint_state(4),'r*');

axes_rob = get(fig_handle,'CurrentAxes');
axes_rob.XLim = [q_min(3), q_max(3)];
axes_rob.YLim = [q_min(4), q_max(4)];

%figure with controls
fig_handle2 = uifigure('Name','Control Panel','Position',[2223 100 450 353]);
c_panel = uipanel(fig_handle2,'Title','Control','FontSize',12,...
        'Position',[25 50 400 300],'Scrollable',1);
joint_idx_names = ["Joint 1","Joint 2",'x','y'];
slider_pos = [100 (1+length(joint_state))*55];
slider_size = [200 3];
sliders = cell(1,length(joint_state));
slabel = sliders;
for i = 1:1:length(joint_state)
    sliders{i} = uislider(c_panel,'Position',[slider_pos-[0 i*50] slider_size],...
         'Limits',[q_min(i) q_max(i)],'Value',q_init(i),'MajorTicks',[q_min(i) q_init(i) q_max(i)],'MinorTicks',linspace(q_min(i),q_max(i),11));
    labeltext = joint_idx_names(i);
    slabel{i} = uilabel(c_panel,'Text',labeltext,'Position', [slider_pos-[90 i*50+20] 80 50]);
    sliders{i}.ValueChangedFcn = @(obj,event)slider_change(obj,event,i, axes_rob,r_h, dot, ctr, cr1, q_min, q_max, y_f); 
    sliders{i}.ValueChangingFcn = @(obj,event)slider_change(obj,event,i, axes_rob,r_h, dot, ctr, cr1, q_min, q_max, y_f); 
end

%% functions 
function slider_change(~, event, i, ax_rob, r_h, dot, ctr, cr1, q_min, q_max, y_f)
    r = [4 4 0];    d = [0 0 0];    alpha = [0 0 0];    base = eye(4);
    pos_enc = @(x)[x sin(x) cos(x)];

    %moving the robot
    global joint_state proj_plot
    joint_state(i) = event.Value;
    move_r(r_h,joint_state(1:2),r,d,alpha,base)
    
    %calculating distance
    tmp = calc_fk(joint_state(1:2),r,d,alpha,base);
    ee_pos = tmp(3,1:2);
%     dst = norm([joint_state(3),joint_state(4)]-ee_pos);
    %dst = y_f(joint_state');

    %info =     sprintf('Pos: %4.2f %4.2f; Dist %4.2f\n',[ee_pos,dst]);
    %fprintf(info);
    
    %moving the point
    dot.XData = joint_state(3);
    dot.YData = joint_state(4);


    proj_plot.XData = joint_state(1);
    proj_plot.YData = joint_state(2);
    if strcmp(event.EventName, 'ValueChanged')
        x_span = linspace(q_min(3),q_max(3));
        y_span = linspace(q_min(4),q_max(4));
        [X_mg,Y_mg] = meshgrid(x_span, y_span);
        x=[X_mg(:) Y_mg(:)]';
        inp = pos_enc([repmat(joint_state(1:2),[length(x),1])'; x]')';
        %val = y_f(inp);
        val = y_f(inp);

        Z_mg = reshape(val,size(X_mg));
        cr1.ZData = Z_mg;
        ctr.ZData = Z_mg;
        ctr.LevelList=linspace(-3,15);
        %contour(X_mg,Y_mg,Z_mg,[1,1.001],'LineStyle','-','LineColor','k','LineWidth',2)

        %contourf(X_mg,Y_mg,Z_mg,100,'LineStyle','none')
    end
    drawnow
end


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

