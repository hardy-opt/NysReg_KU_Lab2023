clear;
close all;

darg = 'IJCNN';
reg = 1e-5;
e = 50; % epoch

dat = strcat('results_NG/',darg);  % result path
data = loaddataa(rng(randperm(1000,1)), 1, 1, dat);

problem = logistic_regressionLM(data.x_train, data.y_train, data.x_test, data.y_test,reg); 


[n,d] = size(data.x_train');
%w = data.w_init;
w = zeros(d,1); x = w; g = problem.full_grad(w); m = ceil(d*0.2);

gx = problem.full_grad(x); x_old = x; xa = 1:1:e; gy = gx; y = x;

cost_x = []; cost_w =[]; Nx = []; Nw =[]; cost_y = []; Ny = [];

Gw = []; Gx = []; Gy = [];

F1= figure; F2= figure; F3= figure;


for i= 1:e
    
 %__________________________    
v = randperm(d,m);
v = sort(v);
S = eye(d);
S = S(:,v);

    % Approximation 1

    g = problem.full_grad(w);
    H = problem.full_hess(w);
    C = H*S;
    M = S'*C;
    gn = norm(g);
    A = C*pinv(M+eye(m)*0)*C';
    Ak = (A+eye(d)*sqrt(gn));
    w = w - (Ak)\g;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gw = [Gw gn];
    c_w = problem.cost(w);
    cost_w = [cost_w c_w];
    nm_w = norm(inv(H) - inv(Ak));
    Nw = [Nw nm_w];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Approximation 2

    gx_old = gx;
    gx = problem.full_grad(x);
    Hx = problem.full_hess(x);
    Cx = Hx*S;
    Mx = S'*Cx;
    sgx = (norm(gx));
    B = Cx*pinv(Mx+eye(m)*0)*Cx';
    x_old = x;
    Bk = (B+eye(d)*(sgx));
    x = x - (Bk\gx);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gx = [Gx sgx];
    c_x = problem.cost(x);
    cost_x = [cost_x c_x];
    nm_x = norm(inv(Hx) - inv(Bk));
    Nx = [Nx nm_x];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Approximation 3
    
    gy_old = gy;
    gy = problem.full_grad(y);
    Hy = problem.full_hess(y);
    Cy = Hy*S;
    My = S'*Cy;
    sgy = (norm(gy));
    F = Cy*pinv(My+eye(m)*0)*Cy';
    y_old = x;
    Fk = (B+eye(d)*(sgy)^2);
    y = y - (Fk\gy);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gy = [Gy sgy];
    c_y = problem.cost(y);
    cost_y = [cost_y c_y];
    nm_y = norm(inv(Hy) - inv(Fk));
    Ny = [Ny nm_y];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    
    figure(F1);
    semilogy(xa(1:i),cost_w,'-r',xa(1:i),cost_x,'--b',xa(1:i),cost_y,'-.g','LineWidth',2);
    hold on;
    
    figure(F2);
    semilogy(xa(1:i),Nw,'-r',xa(1:i),Nx,'--b',xa(1:i),Ny,'-.g','LineWidth',2);
    hold on;
    
    figure(F3);
    semilogy(xa(1:i),Gw,'-r',xa(1:i),Gx,'--b',xa(1:i),Gy,'-.g','LineWidth',2);
    hold on;
end

figure(F1)
legend({'sqrt-Red','norm-blue','square-Green'},'Location','southeast')
ax = gca;
ax.FontSize = 13;
xlabel('Epoch/ Iterations') 
ylabel('Cost') 
title('Cost of Raw and Normalized')


figure(F2)
legend({'sqrt-Red','norm-blue','square-Green'},'Location','southeast')
ax = gca;
ax.FontSize = 13;
xlabel('Epoch/ Iterations') 
ylabel('|| invH - invB ||') 
title('Norm of Hessian with Raw/Normalized App')


figure(F3)
legend({'sqrt-Red','norm-blue','square-green'},'Location','southeast')
ax = gca;
ax.FontSize = 13;
xlabel('Epoch/ Iterations') 
ylabel('|| G ||') 
title('Norm of Gradient')


Info =[];
Info.epoch = (1:e)';
Info.cost_w = cost_w'; % cost 
Info.Nw = Nw'; % Norm of | H - N |
Info.Gw = Gw'; % Norm Gradient

Info.cost_x = cost_x';
Info.Nx = Nx';
Info.Gx = Gx';

Info.cost_y = cost_y';
Info.Ny = Ny';
Info.Gy = Gy';

Name = sprintf('%s/%s_comp.mat','/home/hardik/Desktop/Nys_LM1',darg);


save(Name,'Info');









function [data]=loaddataa(s,reg,step,dat)
    strs = strsplit(dat,'/');
    if strcmp(strs{end}, 'REALSIM')
        data = REALSIM(s,reg,step);
    elseif strcmp(strs{end}, 'CIFAR10B')
        data = CIFAR10B(s,reg,step);
    elseif strcmp(strs{end}, 'MNISTB')
        data = MNISTB(s,reg,step);
    elseif strcmp(strs{end}, 'EPSILON')
        data = EPSILON(s,reg,step);
    elseif strcmp(strs{end}, 'ADULT')
        data = ADULT(s,reg,step);
    elseif strcmp(strs{end}, 'W8A')
        data = W8A(s,reg,step);
    elseif strcmp(strs{end}, 'ALLAML')
        data = ALLAML(s,reg,step);
    elseif strcmp(strs{end}, 'SMK_CAN')
        data = SMK_CAN(s,reg,step);
    elseif strcmp(strs{end}, 'GISETTE')
        data = GISETTE(s,reg,step);
    elseif strcmp(strs{end}, 'A8AN')
        data = A8A(s,reg,step);
    elseif strcmp(strs{end}, 'MRI')
    data = MRI(s);
    elseif strcmp(strs{end}, 'IJCNN')
    data = IJCNN(s,reg,step);
    else
        disp('Dataset tho de');
    end
end
