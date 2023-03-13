clear;
close all;

darg = 'ADULT';
reg = 0;
e = 130; % epoch
times = 1;



AvGw = []; AvGx = []; AvGy = []; AvGp = [0]; AvGq = [0]; AvGr = [0];
AvNw = []; AvNx = []; AvNy = []; AvNp = []; AvNq = []; AvNr = [];

F2= figure; F3= figure; %F4= figure;


dat = strcat('NORM_DIFF/',darg);  % result path

%No of times

for j=1:times

data = loaddataa(j, 1, 1, dat);
%w = data.w_init;

problem = logistic_regressionLM(data.x_train, data.y_train, data.x_test, data.y_test,reg); 

[n,d] = size(data.x_train');

updated = 1;
update=1;%ceil(d*0.05);
m=1;
mcolsr = m;
xa = 1:1:e+1; % e is max epoch

w = zeros(d,1); g = problem.full_grad(w);  % for w nystrom sqrt
 
x = w; gx = g; x_old = x;   % for x mystrom || ||

y = w; gy = g; % for y nystrom with square norm

p = w; gp = g; % for p simple nystrom
 
q = w; gq = g; % for q randomized subspace newton

r = w; gr = g; % for r Newton sketch

cost_x = []; cost_w =[];  cost_y = []; cost_p = []; cost_q = []; cost_r = [];


Gw = []; Gx = []; Gy = []; Gp = [0]; Gq = [0]; Gr = [0];

twn =0; txn=0; tyn=0; tpn=0; tqn=0; trn=0;

%F1= figure;

mvec = [m];

ro = 1/2; cc = 1e-4;
 
delta = 1; % multiplied of hessian regularizer

% g = problem.full_grad(w);
H = problem.full_hess(w);
roh=rank(H)
Hnrm = norm(H);
% Hinv = inv(H);
% Hinrm = norm(Hinv);
gn = sqrt(norm(g));
% H = inv(H);    

Nw = []; Nx = []; Ny = []; Np = []; Nq = []; Nr = [];

Diago = eye(d);

        for i= 1:e


            if j==times && i==1
                figure(F3);
%                   plot(xa(1:i),Gp,'-.pk',xa(1:i),Gr,'--sc','LineWidth',1.5);
%                 plot(xa(1:i),Gp,'-.pk',xa(1:i),Gq,'--om',xa(1:i),Gr,'--sc','LineWidth',1.5);
              plot(m,Gp,'-.pk',m,Gq,'--om',m,Gr,'--sc','LineWidth',1.5);

                hold on; 
            end
         %__________________________    

            % Approximation 1 SQRT



            %C = H*S;
            %M = S'*C;
        %     gn = norm(g);
            %A = C*pinv(M)*C';
            %Ak = (A+eye(d)*sqrt(gn)*delta);
        %     wdir = Ak\g;
        %     step = backtracking_line_search(problem, -wdir, w, ro, cc);
        %     
        %     %%%
        %     c_w = problem.cost(w);
        %     cost_w = [cost_w c_w];
        %     %%%
        %         
        %     w = w - step*wdir;
        %     
        %     
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Gw = [Gw gn];
        %     nm_w = norm((H) - (Ak));
        %     Nw = [Nw nm_w];
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %       tic;
        %       [Z,~,~] = problem.app_hess(w,1:n,v,0);
        %       Awk = Z*Z'+sqrt(gn)*eye(d);
        %       tw= toc;
        %       twn = twn+tw;
        %       Nw =[Nw norm(H-Awk)];
        %       Gw = [Gw twn];

            % Approximation 2 Nystrom with norm of grad

        %     gx_old = gx;
        %     gx = problem.full_grad(x);
        %     Hx = problem.full_hess(x);
        %     Cx = Hx*S;
        %     Mx = S'*Cx;
        %     sgx = (norm(gx));
        %     B = Cx*pinv(Mx)*Cx';
        %     x_old = x;
        %     Bk = (B+eye(d)*(sgx)*delta);
        %     xdir = Bk\gx;
        %     stepx = backtracking_line_search(problem, -xdir, x, ro, cc);
        %     
        %     %%%
        %     c_x = problem.cost(x);
        %     cost_x = [cost_x c_x];
        %     %%%
        %     
        %     
        %     x = x - stepx*xdir ;
        %     
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Gx = [Gx sgx];
        %     nm_x = norm(Hx - Bk);
        %     Nx = [Nx nm_x];
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %       tic;
        %       [Z,~,~] = problem.app_hess(w,1:n,v,0);
        %       Axk = Z*Z'+(gn)*eye(d)*delta;
        %       tx=toc;
        %       txn = txn+tx;
        %       Nx =[Nx norm(H-Axk)];
        %       Gx = [Gx txn];

            % Approximation 3 Nystrom with square of gradiet norm

        %     gy_old = gy;
        %     gy = problem.full_grad(y);
        %     Hy = problem.full_hess(y);
        %     Cy = Hy*S;
        %     My = S'*Cy;
        %     sgy = (norm(gy));
        %     F = Cy*pinv(My+eye(m)*0)*Cy';
        %     y_old = x;
        %     Fk = (F+eye(d)*(sgy)^2*delta);
        %     ydir = Fk\gy;
        %     stepy = backtracking_line_search(problem, -ydir, y, ro, cc);
        %     
        %     %%%
        %     c_y = problem.cost(y);
        %     cost_y = [cost_y c_y];
        %     %%%
        %     
        %     y = y - stepy*ydir;
        %     
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Gy = [Gy sgy];
        % 
        %     nm_y = norm((Hy) - (Fk));
        %     Ny = [Ny nm_y];
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %       tic;
        %       [Z,~,~] = problem.app_hess(w,1:n,v,0);
        %       Ayk = Z*Z'+(gn^2)*eye(d)*delta;
        %       ty=toc;
        %       tyn = tyn+ty;
        %       Ny =[Ny norm(H-Ayk)];
        %       Gy = [Gy tyn];


            % Approximation 4  Nystrom simple

        %     gp = problem.full_grad(p);
        %     Hp = problem.full_hess(p);
        %     Cp = Hp*S;
        %     Mp = S'*Cp;
        %     gpn = norm(gp);
        %     Ap = Cp*pinv(Mp)*Cp';
            %Apk = (Ap+eye(d)*sqrt(gp)*delta);
        %     pdir = pinv(Ap)*gp;
        %     stepp = backtracking_line_search(problem, -pdir, p, ro, cc);
        %     
        %     %%%
        %     c_p = problem.cost(p);
        %     cost_p = [cost_p c_p];
        %     %%%
        %         
        %     p = p - stepp*pdir;
        %     
        %     
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Gp = [Gp gpn];
        %     nm_p = norm((Hp) - (Ap));
        %     Np = [Np nm_p];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            rng(i*j)
            v = randperm(d,m);
            v = sort(v);

            V = eye(d);
            S = V(:,v);


            tic;
             [Z,~,msv] = problem.app_hess(w,1:n,v,0);
             Apk = Z*Z';
            tp=toc;
            rz=rank(Z);
            fprintf('rank(Z)=%d, m=%d\n',rz,m);
            tpn = tpn+tp;
%             if rz==roh
%                 flag=j;
%             end
            Np =[Np (norm(H - (Apk)))];
            Gp = [Gp tpn];
%             PNr = pinv(H*S)';
%             Hinvy= eye(d);
%             RatioNy = norm(Hinv - Hinvy);
%             Ny =[Ny RatioNy];





           

            %%%%%%%%%%%%
            %Prepare Hadamard
        %     Sh = hadamard(m);
        %     Sh = Sh(:,m);
        %Sh = S;
            %%%%%%%%%%%

            % Approximation 5 % Randomized Subspace Newton
        %     
        %     gq = problem.full_grad(q);
        %     Hq = problem.full_hess(q);
            %Cq = Hq*S;
            %Mq = S'*Cq;
            %gqn = norm(gq);
%             Hsqrq = Sh'*problem.hess_sqrt(w,1:n);
%             Mq = Hsqrq*Hsqrq'+reg*Sh'*Sh;
%             %Mq = problem.randomized_sub(w,1:n,v);
% 
%             Aq = (Sh*pinv(Mq)*Sh');
            %Ak = (A+eye(d)*sqrt(gn)*delta);
        %     qdir = Aq*gq;
        %     stepq = backtracking_line_search(problem, -qdir, q, ro, cc);
        %     
        %     %%%
        %     c_q = problem.cost(q);
        %     cost_q = [cost_q c_q];
        %     %%%
        %         
        %     q = q - stepq*qdir;
        %     
        %     
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Gq = [Gq gqn];
        %     nm_q = norm((Hq) - (Aq));
        %     Nq = [Nq nm_q];
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              
        Vrk = eye(n);
        %mm=sort(randperm(n,4*m));
        Srk = Vrk(v,:);  
        
             tic; 
                Hsrk = problem.hess_sqrt(w,1:n)';
                prodrk = (Srk*Hsrk)';
                Ark = (prodrk*prodrk' + eye(d)*reg);
             tq=toc;
             tqn = tqn+tq;
             Nq =[Nq (norm(H - Ark))];
             Gq = [Gq tqn];

            % Approximation 6 %%% Newton sketch

              Vr = eye(n);
              mcols = 4*m;
              mm=sort(randperm(n,mcolsr));
              
              Sr = Vr(mm,:);   


            tic;

        %     vr = randperm(n,8*m);
        %     vr = sort(vr);
        %     Sr = eye(n);
        %     Sr = Sr(vr,:);

        %       if i==1
        %       pow = log(n)/log(2);
        %       
        %       Vr = hadamard(2^ceil(pow));
        %       end
        %     

        %     
        %     gr = problem.full_grad(r);
        %     Hr = problem.full_hess(r);
            %S = S';
        %     grn = norm(gr);
            Hsr = problem.hess_sqrt(w,1:n)';
            prodr = (Sr*Hsr)';
            Ark = (prodr*prodr' + eye(d)*reg);

            tr=toc;
        %     rdir = Ark\g;
        %     stepr = backtracking_line_search(problem, -rdir, r, ro, cc);
        %     
        %     %%%
        %     c_r = problem.cost(r);
        %     cost_r = [cost_r c_r];
        %     %%%
        %         
        %     r = r - stepr*rdir;
        %     
        %     
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     Gr = [Gr grn];
        %     nm_r = norm((Hr) - (Ark));
        %     Nr = [Nr nm_r];
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

              trn = trn+tr;
              Nr =[Nr (norm(H - (Ark)))];
              Gr = [Gr trn];
        %     
        %     figure(F1);
        %     semilogy(xa(1:i),cost_w,'-r',xa(1:i),cost_x,'--b',xa(1:i),cost_y,'-.g',xa(1:i),cost_p,'-.pk',xa(1:i),cost_q,'--om',xa(1:i),cost_r,'-sc','LineWidth',2);
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

            m = m+update;

            
            
            if m>d || m==d
                m=d-1;
            end
            
%             if flag==j
%                 m=d;
%             end

            if 4*m>n %8*
                mcolsr= n;%m-update;
            else
                mcolsr = mcols;
            end
            
            mvec = [mvec m];
            fprintf('epoch= i =%d, and runtime = j = %d\n',i,j);
        end

% figure(F1)
% legend({'sqrt-Red','norm-blue','square-Green','Nystrom','RSN','NS'},'Location','southeast')
% ax = gca;
% ax.FontSize = 13;
% xlabel('t (iterations)') 
% ylabel('cost') 
% title('cost of Raw and Normalized')

    if j==1
       AvNp = Np;  AvNr=Nr; AvNq=Nq;
    AvGp = Gp; AvGq =  Gq; AvGr = Gr;
        AvNy = Ny;
    end


    AvNp = AvNp + Np; AvNr = AvNr+Nr; AvNq = AvNq+Nq;
    AvGp = AvGp + Gp; AvGq = AvGq + Gq; AvGr = AvGr + Gr;
    AvNy = AvNy+Ny;

end

if j==times
   ji=j+1; 
    AvNp = (1/ji)*AvNp; AvNr = (1/ji)*AvNr; AvNq = (1/ji)*AvNq;
    AvGp = (1/ji)*AvGp; AvGq = (1/ji)*AvGq; AvGr = AvGr*(1/ji);

            figure(F2);
%             semilogy(xa(1:i),AvNp,'-.pk',xa(1:i),AvNq,'--om',xa(1:i),AvNr,'-sc','LineWidth',1.5);
            semilogy(mvec(:,1:end-1),AvNp,'-.pk',mvec(:,1:end-1),AvNq,'--om',mvec(:,1:end-1),AvNr,'-sc','LineWidth',1.8);

            hold on;

% 
%             figure(F3);
%             semilogy(xa(1:i+1),AvGp,'-.pk',xa(1:i+1),AvGr,'--sc','LineWidth',1.5);
%             hold on;

            figure(F3);
            semilogy(mvec,AvGp,'-.pk',mvec,AvGq,'--om',mvec,AvGr,'--sc','LineWidth',1.8);
            hold on;

% 
%             figure(F4);
%             semilogy(xa(1:i),AvNy,'-.pk',xa(1:i),AvNq,'--om','LineWidth',1.5);
%             hold on;




figure(F2)
% xticklabels({'1',' ','1000',' ','2000',' ','3000','','d'})
legend({'Nystrom','NS','NS(4m)'},'Location','southeast')
ax = gca;
ax.FontSize = 13;
xlabel('m-columns') 
ylabel('$\| H - B \|$','interpreter','latex') 
%title('$\lambda = 1e-5$','interpreter','latex');

figure(F3)
legend({'Nystrom','NS','NS(4m)'},'Location','southeast')
% xticklabels({'1','1000','2000','3000','4000','5000'})
ax = gca;
ax.FontSize = 13;
xlabel('m-columns') 
ylabel('CPU time') 
%title('$\lambda = 1e-5$','interpreter','latex');
% 
% figure(F4)
% % xticklabels({'1','1000','2000','3000','4000','5000'})
% legend({'Nystrom','RSN'},'Location','southeast')
% ax = gca;
% ax.FontSize = 13;
% xlabel('m-rank') 
% ylabel('$\| H - B \|$','interpreter','latex') 
% title('norm of Hessian difference');

end

Info =[];
Info.epoch = (1:e)';
Info.cost_w = cost_w'; % cost 
Info.Nw = Nw'; % Norm of | H - N |
Info.Gw = Gw'; % Norm Gradient

Info.cost_x = cost_x';
Info.AvNx = AvNx';
Info.AvGx = AvGx';

Info.cost_y = cost_y';
Info.AvNy = AvNy';
Info.AvGy = AvGy';


Info.cost_p = cost_p';
Info.AvNp = AvNp';
Info.AvGp = AvGp';

Info.cost_q = cost_q';
Info.AvNq = AvNq';
Info.AvGq = AvGq';

Info.cost_r = cost_r';
Info.AvNr = AvNr';
Info.AvGr = AvGr';
Info.mvec = mvec';
Name = sprintf('%s/%s_NS_comp_%.d.mat','/home/optima/Desktop/Nys_Newton23',darg,reg);


save(Name,'Info');

%If you lose FIGURE, load the save file and RUN (figure 2) Norm diff >RUN
%semilogy(Info.mvec(2:end,:),Info.AvNp,'-.pk',Info.mvec(2:end,:),Info.AvNq,'--om',Info.mvec(2:end,:),Info.AvNr,'-sc','LineWidth',1.5);
%For Figure 3
%semilogy(Info.mvec,Info.AvGp,'-.pk',Info.mvec,Info.AvGq,'--om',Info.mvec,Info.AvGr,'--sc','LineWidth',1.5);
%and do run title and legends



% clear;
% close all;
% 
% darg = 'W8A'; %'GISETTE';
% reg = 1e-5;
% e = 10; % epoch
% 
% dat = strcat('NORM_DIFF/',darg);  % result path
% data = loaddataa(rng(randperm(100,1)), 1, 1, dat);
% %w = data.w_init;
% 
% problem = logistic_regressionLM(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
% 
% [n,d] = size(data.x_train');
% 
% %m = ceil(d*0.2);
% m=1;
% xa = 1:1:e; % e is max epoch
% 
% w = zeros(d,1); g = problem.full_grad(w);  % for w nystrom sqrt
%  
% x = w; gx = g; x_old = x;   % for x mystrom || ||
% 
% y = w; gy = g; % for y nystrom with square norm
% 
% p = w; gp = g; % for p simple nystrom
%  
% q = w; gq = g; % for q randomized subspace newton
% 
% r = w; gr = g; % for r Newton sketch
% 
% cost_x = []; cost_w =[];  cost_y = []; cost_p = []; cost_q = []; cost_r = [];
% 
% Nw = []; Nx = []; Ny = []; Np = []; Nq = []; Nr = [];
% 
% Gw = []; Gx = []; Gy = []; Gp = []; Gq = []; Gr = [];
% 
% twn =0; txn=0; tyn=0; tpn=0; tqn=0; trn=0;
% 
% %F1= figure;
% F2= figure; F3= figure;
% 
% ro = 1/2; cc = 1e-4;
%  
% delta = 1; % multiplied of hessian regularizer
% 
% g = problem.full_grad(w);
% H = problem.full_hess(w);
%     
% 
% for i= 1:e
%     
%  %__________________________    
% v = randperm(d,m);
% size(v)
% v = sort(v);
% S = eye(d);
% S = S(:,v);
% 
%     % Approximation 1 SQRT
% 
% 
%     
%     %C = H*S;
%     %M = S'*C;
%     gn = norm(g);
%     %A = C*pinv(M)*C';
%     %Ak = (A+eye(d)*sqrt(gn)*delta);
% %     wdir = Ak\g;
% %     step = backtracking_line_search(problem, -wdir, w, ro, cc);
% %     
% %     %%%
% %     c_w = problem.cost(w);
% %     cost_w = [cost_w c_w];
% %     %%%
% %         
% %     w = w - step*wdir;
% %     
% %     
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %     Gw = [Gw gn];
% %     nm_w = norm((H) - (Ak));
% %     Nw = [Nw nm_w];
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%       tic;
%       [Z,~,~] = problem.app_hess(w,1:n,v,0);
%       Awk = Z*Z'+sqrt(gn)*eye(d);
%       tw= toc;
%       twn = twn+tw;
%       Nw =[Nw norm(H-Awk)];
%       Gw = [Gw twn];
% 
%     % Approximation 2 Nystrom with norm of grad
%    
% %     gx_old = gx;
% %     gx = problem.full_grad(x);
% %     Hx = problem.full_hess(x);
% %     Cx = Hx*S;
% %     Mx = S'*Cx;
% %     sgx = (norm(gx));
% %     B = Cx*pinv(Mx)*Cx';
% %     x_old = x;
% %     Bk = (B+eye(d)*(sgx)*delta);
% %     xdir = Bk\gx;
% %     stepx = backtracking_line_search(problem, -xdir, x, ro, cc);
% %     
% %     %%%
% %     c_x = problem.cost(x);
% %     cost_x = [cost_x c_x];
% %     %%%
% %     
% %     
% %     x = x - stepx*xdir ;
% %     
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %     Gx = [Gx sgx];
% %     nm_x = norm(Hx - Bk);
% %     Nx = [Nx nm_x];
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%       tic;
%       [Z,~,~] = problem.app_hess(w,1:n,v,0);
%       Axk = Z*Z'+(gn)*eye(d)*delta;
%       tx=toc;
%       txn = txn+tx;
%       Nx =[Nx norm(H-Axk)];
%       Gx = [Gx txn];
% 
%     % Approximation 3 Nystrom with square of gradiet norm
%         
% %     gy_old = gy;
% %     gy = problem.full_grad(y);
% %     Hy = problem.full_hess(y);
% %     Cy = Hy*S;
% %     My = S'*Cy;
% %     sgy = (norm(gy));
% %     F = Cy*pinv(My+eye(m)*0)*Cy';
% %     y_old = x;
% %     Fk = (F+eye(d)*(sgy)^2*delta);
% %     ydir = Fk\gy;
% %     stepy = backtracking_line_search(problem, -ydir, y, ro, cc);
% %     
% %     %%%
% %     c_y = problem.cost(y);
% %     cost_y = [cost_y c_y];
% %     %%%
% %     
% %     y = y - stepy*ydir;
% %     
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %     Gy = [Gy sgy];
% % 
% %     nm_y = norm((Hy) - (Fk));
% %     Ny = [Ny nm_y];
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       tic;
%       [Z,~,~] = problem.app_hess(w,1:n,v,0);
%       Ayk = Z*Z'+(gn^2)*eye(d)*delta;
%       ty=toc;
%       tyn = tyn+ty;
%       Ny =[Ny norm(H-Ayk)];
%       Gy = [Gy tyn];
% 
%     
%     % Approximation 4  Nystrom simple
%     
% %     gp = problem.full_grad(p);
% %     Hp = problem.full_hess(p);
% %     Cp = Hp*S;
% %     Mp = S'*Cp;
% %     gpn = norm(gp);
% %     Ap = Cp*pinv(Mp)*Cp';
%     %Apk = (Ap+eye(d)*sqrt(gp)*delta);
% %     pdir = pinv(Ap)*gp;
% %     stepp = backtracking_line_search(problem, -pdir, p, ro, cc);
% %     
% %     %%%
% %     c_p = problem.cost(p);
% %     cost_p = [cost_p c_p];
% %     %%%
% %         
% %     p = p - stepp*pdir;
% %     
% %     
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %     Gp = [Gp gpn];
% %     nm_p = norm((Hp) - (Ap));
% %     Np = [Np nm_p];
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     tic;
%     [Z,~,~] = problem.app_hess(w,1:n,v,0);
%     Apk = Z*Z';
%     tp=toc;
%     tpn = tpn+tp;
%     Np =[Np norm(H-Apk)];
%     Gp = [Gp tpn];
%     
%     
%     tic;
%     % Approximation 5 % Randomized Subspace Newton
% %     
% %     gq = problem.full_grad(q);
% %     Hq = problem.full_hess(q);
%     %Cq = Hq*S;
%     %Mq = S'*Cq;
%     %gqn = norm(gq);
%     Mq = problem.randomized_sub(w,1:n,v);
%     Aq = (S*Mq*S');
%     %Ak = (A+eye(d)*sqrt(gn)*delta);
% %     qdir = Aq*gq;
% %     stepq = backtracking_line_search(problem, -qdir, q, ro, cc);
% %     
% %     %%%
% %     c_q = problem.cost(q);
% %     cost_q = [cost_q c_q];
% %     %%%
% %         
% %     q = q - stepq*qdir;
% %     
% %     
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %     Gq = [Gq gqn];
% %     nm_q = norm((Hq) - (Aq));
% %     Nq = [Nq nm_q];
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%      tq=toc;
%      tqn = tqn+tq;
%      Nq =[Nq norm(H-Aq)];
%      Gq = [Gq tqn];
% 
%     % Approximation 6 %%% Newton sketch
%     
%     tic;
%     
%     vr = randperm(n,8*m);
%     vr = sort(vr);
%     Sr = eye(n);
%     Sr = Sr(vr,:);
%     size(Sr)
% %     
% %     gr = problem.full_grad(r);
% %     Hr = problem.full_hess(r);
%     %S = S';
% %     grn = norm(gr);
%     Hsr = problem.hess_sqrt(w,1:n);
%     prodr = Hsr*Sr';
%     Ark = (prodr*prodr' + eye(d)*reg);
%     
%     tr=toc;
% %     rdir = Ark\g;
% %     stepr = backtracking_line_search(problem, -rdir, r, ro, cc);
% %     
% %     %%%
% %     c_r = problem.cost(r);
% %     cost_r = [cost_r c_r];
% %     %%%
% %         
% %     r = r - stepr*rdir;
% %     
% %     
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %     Gr = [Gr grn];
% %     nm_r = norm((Hr) - (Ark));
% %     Nr = [Nr nm_r];
% %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%       trn = trn+tr;
%       Nr =[Nr norm(H-Ark)];
%       Gr = [Gr trn];
% %     
% %     figure(F1);
% %     semilogy(xa(1:i),cost_w,'-r',xa(1:i),cost_x,'--b',xa(1:i),cost_y,'-.g',xa(1:i),cost_p,'-.pk',xa(1:i),cost_q,'--om',xa(1:i),cost_r,'-sc','LineWidth',2);
% %     hold on;
% %     
%     figure(F2);
%     semilogy(xa(1:i),Nw,'--r',xa(1:i),Nx,'.b',xa(1:i),Ny,'-.g',xa(1:i),Np,'-.pk',xa(1:i),Nq,'--om',xa(1:i),Nr,'-sc','LineWidth',1.5);
%     hold on;
%     
%     figure(F3);
%     semilogy(xa(1:i),Gw,'--r',xa(1:i),Gx,'.b',xa(1:i),Gy,'-.g',xa(1:i),Gp,'-.pk',xa(1:i),Gq,'--om',xa(1:i),Gr,'--sc','LineWidth',1.5);
%     hold on;
%     
% %     
% %     figure(F1);
% %     semilogy(xa(1:i),cost_w,'-r',xa(1:i),cost_x,'--b',xa(1:i),cost_y,'-.g','LineWidth',2);
% %     hold on;
% %     
% %     figure(F2);
% %     semilogy(xa(1:i),Nw,'-r',xa(1:i),Nx,'--b',xa(1:i),Ny,'-.g','LineWidth',2);
% %     hold on;
% %     
% %     figure(F3);
% %     semilogy(xa(1:i),Gw,'-r',xa(1:i),Gx,'--b',xa(1:i),Gy,'-.g','LineWidth',2);
% %     hold on;
%   
%     m = m+50;
%     if m>d
%         m=d;
%     end
%     
% end
% 
% % figure(F1)
% % legend({'sqrt-Red','norm-blue','square-Green','Nystrom','RSN','NS'},'Location','southeast')
% % ax = gca;
% % ax.FontSize = 13;
% % xlabel('t (iterations)') 
% % ylabel('cost') 
% % title('cost of Raw and Normalized')
% 
% 
% figure(F2)
% legend({'sqrt-Red','norm-blue','square-Green','Nystrom','RSN','NS'},'Location','southeast')
% ax = gca;
% ax.FontSize = 13;
% xlabel('m-rank') 
% ylabel('$\| H - B \|$','interpreter','latex') 
% title('norm of Hessian difference')
% 
% 
% figure(F3)
% legend({'sqrt-Red','norm-blue','square-green','Nystrom','RSN','NS'},'Location','southeast')
% ax = gca;
% ax.FontSize = 13;
% xlabel('m-rank') 
% ylabel('CPU time') 
% title('CPU time comparison')
% 
% 
% Info =[];
% Info.epoch = (1:e)';
% Info.cost_w = cost_w'; % cost 
% Info.Nw = Nw'; % Norm of | H - N |
% Info.Gw = Gw'; % Norm Gradient
% 
% Info.cost_x = cost_x';
% Info.Nx = Nx';
% Info.Gx = Gx';
% 
% Info.cost_y = cost_y';
% Info.Ny = Ny';
% Info.Gy = Gy';
% 
% 
% Info.cost_p = cost_p';
% Info.Np = Np';
% Info.Gp = Gp';
% 
% Info.cost_q = cost_q';
% Info.Nq = Nq';
% Info.Gq = Gq';
% 
% Info.cost_r = cost_r';
% Info.Nr = Nr';
% Info.Gr = Gr';
% 
% Name = sprintf('%s/%s_CPU.mat','/home/optima/Desktop/Nys_Newton23',darg);
% 
% 
% save(Name,'Info');
% 
% 
% 
% 
% 




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
    elseif strcmp(strs{end}, 'RCV1')
    data = RCV1(s,reg,step);
    elseif strcmp(strs{end}, 'COLONCANCER')
        data = COLONCANCER(s,reg,step);
    else
        disp('Dataset tho de');
    end
end
