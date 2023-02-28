data = ADULT(1,1,1);%W8A(1,1,1);
mu = [ 1e-3 1e-2 1e-1 1e-0 1e1 1e2 1e3 ];
s = [-3:1:3];
l = length(mu);
eff_dim = zeros(l,1);
[n,d] = size(data.x_train');
rng(1);
w = randn(d,1);
rng(4);
w1 = rand(d,1);
n = 5000;
data.x_train = data.x_train(:,1:n);
data.y_train = data.y_train(1:n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eff_dimr = zeros(l,1);
eff_dimny = zeros(l,1);
eff_dimny1 = zeros(l,1);
eff_dimr1 = zeros(l,1);
eff_dimr2 = zeros(l,1);
nd = zeros(l,1);
nd2 = zeros(l,1);
nd1 = zeros(l,1);
ndny = zeros(l,1);
ndny1 = zeros(l,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:l
rng(l)
w = ones(d,1);
problem = logistic_regression1N(data.x_train, data.y_train, data.x_test, data.y_test,mu(i));
H1 = problem.full_hess(w);
H = (1/n)*H1;
rank(H)
eff_dim(i) = trace(H*inv(H + mu(i)*eye(d)));
H = H +  mu(i)*eye(d);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = 1:100;
p1 = 2.^p;
T = floor(p1./n);
ind = find(T==1);
Had = hadamard(2^ind);
S = Had(1:eff_dim(i),1:n);
Z = problem.hess_sqrt(w,1:n);
Hs = (1/n)*Z*S'*S*Z'+  mu(i)*eye(d);
eff_dimr(i) = rank(Hs);
hn = norm(H+mu(i)*eye(d),'fro');
nd(i) = norm(H - Hs,'fro')/hn;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S1 = Had(1:10*eff_dim(i),1:n);
Z = problem.hess_sqrt(w,1:n);
Hs1 = (1/n)*Z*S1'*S1*Z';
eff_dimr1(i) = rank(Hs1);
%hn = norm(H+mu(i)*eye(301),'fro');
nd1(i) = norm(H - Hs1,'fro')/hn;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S2 = Had(1:20*eff_dim(i),1:n);
Z = problem.hess_sqrt(w,1:n);
Hs2 = (1/n)*Z*S2'*S2*Z';
eff_dimr2(i) = rank(Hs2);
%hn = norm(H+mu(i)*eye(301),'fro');
nd2(i) = norm(H - Hs2,'fro')/hn;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v = randperm(d,10);
m = zeros(d,1);
m(v) = 1;
M = diag(m);
V = M(:,v);
%C = Z'*V;
%[U,f,A] = svd(C);
%I = U(:,1:10);
%Ny = Z*I*I'*Z';
C=(1/n)*H1*V;
W = C(v,:);
[U,f,A] = svd(W);
r = rank(f);
fin = inv(f(1:r,1:r));
U = U(:,1:r);
Ny = C*U*fin*U'*C'+mu(i)*eye(d);
eff_dimny(i) = rank(Ny);
ndny(i) = norm(H - Ny,'fro')/hn;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v1 = v;%randperm(d,20);
m1 = zeros(d,1);
m1(v1) = 1;
M1 = diag(m1);
V1 = M1(:,v1);
C1=H1*V1;
W1 = C1(v1,:);
[U1,f1,A1] = svd(W1);
r = rank(f1);
fin1 = inv(f1(1:r,1:r));
U1 = U1(:,1:r);
Ny1 =(1/n)*(C1*U1*fin1*U1'*C1')+mu(i)*eye(d);
eff_dimny1(i) = rank(Ny1);
ndny1(i) = norm(H - Ny1,'fro')/hn;
end
plot(s,eff_dimr,'--o',s,eff_dimny,'--*',s,eff_dimny1,'--p','LineWidth',2);
%plot(s,eff_dimr,'--o',s,eff_dimr1,'--*',s,eff_dimr2,'--s',s,eff_dimny,'--p','LineWidth',1.5);
title('Effective Dimension')
set(gca, 'YScale', 'log')
xlabel('lambda-log scale')
xticks([ -3 -2 -1 0 1 2 3 ]);
xticklabels({'10^{-3}','10^{-2}','10^{-1}','10^0','10^1','10^2','10^3'});
ylabel('rank')
legend('rank(Hs) at m = eff(d)','rank(Nys) at m = 10', 'rank(Nys1) at m = 10')
%legend('rank(Hs) at m = eff(d)','rank(Hs1) at m = 10*eff(d)', 'rank(Hs2) at m = 20*eff(d)', 'rank(Nys) at m = 10')
figure

plot(nd,'--o'); hold on;
plot(ndny,'--*'); hold on;
plot(ndny1,'--p','LineWidth',2)
%set(gca, 'YScale', 'log')
xlabel('lambda-log scale')
%xticks([ -3 -2 -1 0 1 2 3 ]);
xticklabels({'10^{-3}','10^{-2}','10^{-1}','10^0','10^1','10^2','10^3'});
ylabel('||H - Approx(H)||')
legend('|H-Hs|/|H|','|H-Nys|/|H|','|H-Nys1|/|H|')
%legend('norm(H-Hs)','norm(H-Hs1)','norm(H-Hs2)','norm(H-Nys)')
%xscale('log')

figure; subplot(1,4,1)
imagesc(H);
title('Hessian H')
subplot(1,4,2)
imagesc(Hs);
title('Hs with m = eff(d)')
subplot(1,4,3)
imagesc(Ny);
title('Hs1 with m = 10*eff(d)')
%subplot(1,5,4)
%imagesc(Hs2);
subplot(1,4,4)
imagesc(Ny1);
title('Nystr\"om with m = 10')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot(s,eff_dimr,'--o',s,eff_dimr1,'--*',s,eff_dimr2,'--s',s,eff_dimny,'--p',s,eff_dimny1,'-->','LineWidth',1.5);
% title('Effective Dimension')
% xlabel('mu-log scale')
% xticks([ -3 -2 -1 0 1 2 3 ]);
% xticklabels({'10^{-3}','10^{-2}','10^{-1}','10^0','10^1','10^2','10^3'});
% ylabel('rank')
% legend('rank(Hs) at m = eff(d)','rank(Hs1) at m = 10*eff(d)', 'rank(Hs2) at m = 20*eff(d)', 'rank(Nys) at m = 10','rank(Nys) at m = 20')
% 
% figure
% plot(s,nd,'--o',s,nd1,'--*',s,nd2,'--s',s,ndny,'--p',s,ndny1,'-->','LineWidth',1.5)
% set(gca, 'YScale', 'log')
% xlabel('mu-log scale')
% xticks([ -3 -2 -1 0 1 2 3 ]);
% xticklabels({'10^{-3}','10^{-2}','10^{-1}','10^0','10^1','10^2','10^3'});
% ylabel('||H - Approx(H)||')
% 
% legend('norm(H-Hs)','norm(H-Hs1)','norm(H-Hs2)','norm(H-Nys)','norm(H-Nys1)')
% %xscale('log')
% 
% figure; subplot(1,6,1)
% imagesc(H);
% subplot(1,6,2)
% imagesc(Hs);
% subplot(1,6,3)
% imagesc(Hs1);
% subplot(1,6,4)
% imagesc(Hs2);
% subplot(1,6,5)
% imagesc(Ny);
% subplot(1,6,6)
% imagesc(Ny1);