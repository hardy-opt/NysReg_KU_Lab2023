clear;
close all;

darg = 'ADULT';
reg = 1e-5;
e = 15; % epoch

dat = strcat('NORM_DIFF/',darg);  % result path
data = loaddataa((randperm(100,1)), 1, 1, dat);
%w = data.w_init;

problem = logistic_regressionLM(data.x_train, data.y_train, data.x_test, data.y_test,reg); 

[n,d] = size(data.x_train');

m = ceil(d*0.1);

xa = 1:1:e+1; % e is max epoch

w = zeros(d,1); g = problem.full_grad(w);  % for w nystrom sqrt
 
x = w; gx = g; x_old = x;   % for x mystrom || ||

y = w; gy = g; % for y nystrom with square norm

p = w; gp = g; % for p simple nystrom
 
q = w; gq = g; % for q randomized subspace newton

r = w; gr = g; % for r Newton sketch

cost_x = []; cost_w =[];  cost_y = []; cost_p = []; cost_q = []; cost_r = [];

NM_H = norm(problem.full_hess(w));

Nw = [NM_H]; Nx = [NM_H]; Ny = [NM_H]; Np = [NM_H]; Nq = [NM_H]; Nr = [NM_H];

Gw = []; Gx = []; Gy = []; Gp = []; Gq = []; Gr = [];

F1= figure; F2= figure; F3= figure;

ro = 1/2; cc = 1e-4;
 
delta = 0.1; % multiplied of hessian regularizer

for i= 1:e
    
 %__________________________    
v = randperm(d,m);
v = sort(v);
S = eye(d);
S = S(:,v);

    % Approximation 1 SQRT

    g = problem.full_grad(w);
    H = problem.full_hess(w);
    C = H*S;
    M = S'*C;
    gn = norm(g);
    A = C*pinv(M)*C';
    Ak = (A+eye(d)*sqrt(gn)*delta);
    wdir = Ak\g;
    step = backtracking_line_search(problem, -wdir, w, ro, cc);
    
    %%%
    c_w = problem.cost(w);
    cost_w = [cost_w c_w];
    %%%
        
    w = w - step*wdir;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gw = [Gw gn];
    nm_w = norm((H) - (Ak));
    Nw = [Nw nm_w];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Approximation 2 Nystrom with norm of grad

    gx_old = gx;
    gx = problem.full_grad(x);
    Hx = problem.full_hess(x);
    Cx = Hx*S;
    Mx = S'*Cx;
    sgx = (norm(gx));
    B = Cx*pinv(Mx)*Cx';
    x_old = x;
    Bk = (B+eye(d)*(sgx)*delta);
    xdir = Bk\gx;
    stepx = backtracking_line_search(problem, -xdir, x, ro, cc);
    
    %%%
    c_x = problem.cost(x);
    cost_x = [cost_x c_x];
    %%%
    
    
    x = x - stepx*xdir ;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gx = [Gx sgx];
    nm_x = norm(Hx - Bk);
    Nx = [Nx nm_x];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Approximation 3 Nystrom with square of gradiet norm
    
    gy_old = gy;
    gy = problem.full_grad(y);
    Hy = problem.full_hess(y);
    Cy = Hy*S;
    My = S'*Cy;
    sgy = (norm(gy));
    F = Cy*pinv(My+eye(m)*0)*Cy';
    y_old = x;
    Fk = (F+eye(d)*(sgy)^2*delta);
    ydir = Fk\gy;
    stepy = backtracking_line_search(problem, -ydir, y, ro, cc);
    
    %%%
    c_y = problem.cost(y);
    cost_y = [cost_y c_y];
    %%%
    
    y = y - stepy*ydir;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gy = [Gy sgy];

    nm_y = norm((Hy) - (Fk));
    Ny = [Ny nm_y];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Approximation 4  Nystrom simple
    
    gp = problem.full_grad(p);
    Hp = problem.full_hess(p);
    Cp = Hp*S;
    Mp = S'*Cp;
    gpn = norm(gp);
    [Up,Sup] =svd((Mp));
    Ap = Cp*Up*inv(Sup)*Up'*Cp';
    %Apk = (Ap+eye(d)*sqrt(gp)*delta);
    pdir = pinv(Ap)*gp;
    stepp = backtracking_line_search(problem, -pdir, p, ro, cc);
    
    %%%
    c_p = problem.cost(p);
    cost_p = [cost_p c_p];
    %%%
        
    p = p - stepp*pdir;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gp = [Gp gpn];
    nm_p = norm((Hp) - (Ap));
    Np = [Np nm_p];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Approximation 5 % Randomized Subspace Newton
    
    gq = problem.full_grad(q);
    Hq = problem.full_hess(q);
    Cq = Hq*S;
    Mq = S'*Cq;
    gqn = norm(gq);
    Aq = S*pinv(Mq)*S';
    %Ak = (A+eye(d)*sqrt(gn)*delta);
    qdir = Aq*gq;
    stepq = backtracking_line_search(problem, -qdir, q, ro, cc);
    
    %%%
    c_q = problem.cost(q);
    cost_q = [cost_q c_q];
    %%%
        
    q = q - stepq*qdir;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gq = [Gq gqn];
    nm_q = norm((Hq) - (Aq));
    Nq = [Nq nm_q];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Approximation 6 %%% Newton sketch
    
    vr = randperm(n,m);
    vr = sort(vr);
    Sr = eye(n);
    Sr = Sr(v,:);
    
    
    gr = problem.full_grad(r);
    Hr = problem.full_hess(r);
    %S = S';
    grn = norm(gr);
    Hsr = problem.hess_sqrt(r,1:n);
    prodr = Hsr*Sr';
    Ark = (prodr*prodr' + eye(d)*reg);
    rdir = Ark\g;
    stepr = backtracking_line_search(problem, -rdir, r, ro, cc);
    
    %%%
    c_r = problem.cost(r);
    cost_r = [cost_r c_r];
    %%%
        
    r = r - stepr*rdir;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gr = [Gr grn];
    nm_r = norm((Hr) - (Ark));
    Nr = [Nr nm_r];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    figure(F1);
    semilogy(xa(1:i),cost_w,'-r',xa(1:i),cost_x,'--b',xa(1:i),cost_y,'-.g',xa(1:i),cost_p,'-.pk',xa(1:i),cost_q,'-->c','LineWidth',2);
    hold on;
    
   
    
    figure(F2);
    semilogy(xa(1:i+1),Nw,'-r',xa(1:i+1),Nx,'--b',xa(1:i+1),Ny,'-.g',xa(1:i+1),Np,'-.pk',xa(1:i+1),Nq,'-->c','LineWidth',2);
    hold on;
     
    
    
    figure(F3);
    semilogy(xa(1:i),Gw,'-r',xa(1:i),Gx,'--b',xa(1:i),Gy,'-.g',xa(1:i),Gp,'-.pk',xa(1:i),Gq,'-->c','LineWidth',2);
    hold on;
    
    
    
%     figure(F1);
%     semilogy(xa(1:i),cost_w,'-r',xa(1:i),cost_x,'--b',xa(1:i),cost_y,'-.g',xa(1:i),cost_p,'-.pk',xa(1:i),cost_q,'--om',xa(1:i),cost_r,'-sc','LineWidth',2);
%     hold on;
%     
%    
%     
%     figure(F2);
%     semilogy(xa(1:i+1),Nw,'-r',xa(1:i+1),Nx,'--b',xa(1:i+1),Ny,'-.g',xa(1:i+1),Np,'-.pk',xa(1:i+1),Nq,'--om',xa(1:i+1),Nr,'-sc','LineWidth',2);
%     hold on;
%      
%     
%     
%     figure(F3);
%     semilogy(xa(1:i),Gw,'-r',xa(1:i),Gx,'--b',xa(1:i),Gy,'-.g',xa(1:i),Gp,'-.pk',xa(1:i),Gq,'--om',xa(1:i),Gr,'-sc','LineWidth',2);
%     hold on;
%     
%     
%     figure(F1);
%     semilogy(xa(1:i),cost_w,'-r',xa(1:i),cost_x,'--b',xa(1:i),cost_y,'-.g','LineWidth',2);
%     hold on;
%     
%     figure(F2);
%     semilogy(xa(1:i),Nw,'-r',xa(1:i),Nx,'--b',xa(1:i),Ny,'-.g','LineWidth',2);
%     hold on;
%     
%     figure(F3);
%     semilogy(xa(1:i),Gw,'-r',xa(1:i),Gx,'--b',xa(1:i),Gy,'-.g','LineWidth',2);
%     hold on;
    
end

figure(F1)
legend({'NGD','NGD1','NGD2','Nystrom','RSN'},'Location','southeast')%'RSN'
ax = gca;
ax.FontSize = 13;
xlabel('t (iterations)') 
ylabel('$f(w)$','interpreter','latex') 
title('cost')


figure(F2)
legend({'NGD','NGD1','NGD2','Nystrom','RSN'},'Location','southeast') %'RSN'
ax = gca;
ax.FontSize = 13;
xlabel('t (iterations)') 
ylabel('$\| H - B \|$','interpreter','latex') 
title('norm of Hessian difference')


figure(F3)
legend({'NGD','NGD1','NGD2','Nystrom','RSN'},'Location','southeast')%'RSN'
ax = gca;
ax.FontSize = 13;
xlabel('t (iterations)') 
ylabel('$\| g \|$','interpreter','latex') 
title('norm of gradient')


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

Name = sprintf('%s/%s_RSN_comp_%.d.mat','/home/optima/Desktop/Nys_Newton23',darg,reg);


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
