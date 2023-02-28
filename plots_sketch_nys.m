clear all;
close all;
addpath(genpath(pwd))
data=W8A(1,1,1);
w=data.w_init;
X=data.x_train';
[n,d]=size(X);
n=5000;
X=X(randperm(49749,n),:);
D=diag(sigmoid(X*w));
H=X'*D*X/n;
H=H/max(max(H));

B=sqrt(D)*X;
rhos =[1e-5,1e-4,1e-3,1e-2,1e-1,1e-0];
dim=zeros(6,1);
errs=zeros(6,1);
errs1=zeros(6,1);
errn=zeros(6,1);
errn1=zeros(6,1);
ranks=zeros(6,1);
ranks1=zeros(6,1);
ranksn=zeros(6,1);
ranksn1=zeros(6,1);

Hss={};
Hss1={};
Hns={};
Hns1={};

for i=1:6
% dim(i)=ceil(trace(H*inv(H+rhos(i)*eye(d))));
% Sr=randn(ceil(dim(i)),n);
% S=orth(Sr')';
% mx=max(max(S'*S));
% Hs=B'*S'*S*B/(n*mx);
% Hs=Hs/max(max(Hs));
% Hss{i}=Hs;
% errs(i)=norm(H-Hs,'fro')/norm(H,'fro');
% ranks(i)=rank(Hs);

% Sr=randn(ceil(10),n);
% S=orth(Sr')';
% mx=max(max(S'*S));
% Hs=B'*S'*S*B/(n*mx);
% Hs=Hs/max(max(Hs));
% Hss1{i}=Hs;
% errs1(i)=norm(H-Hs,'fro')/norm(H,'fro');
% ranks1(i)=rank(Hs);

idx=randperm(d-1,10);
C=H(:,idx);
M=C(idx,:);
[u,s,v]=svd(M);
r=rank(s);
u=u(:,1:r);
s=diag(1./diag(s(1:r,1:r)));
Mi=u*s*u';
%Mi=Mi/max(max(Mi));
Hn=C*Mi*C';
Hn=Hn/max(max(Hn));
Hns{i}=Hn;
errn(i)=norm(H-Hn,'fro')/norm(H,'fro');
ranksn(i)=rank(Hn);

%%%%
Q = Hn;
Hs = (inv(Q+eye(d)*rhos(i)*10))*(Q*Q');%+rhos(i)*eye(d));
%Hs=B'*S'*S*B/(n*mx);
Hs=Hs/max(max(Hs));
Hss{i}=Hs;
errs(i)=norm(H-Hs,'fro')/norm(H,'fro');
ranks(i)=rank(Hs);

%%%%%


idx=randperm(d-1,20);
C=H(:,idx);
M=C(idx,:);
[u,s,v]=svd(M);
r=rank(s);
u=u(:,1:r);
s=diag(1./diag(s(1:r,1:r)));
Mi=u*s*u';
%Mi=Mi/max(max(Mi));
Hn=C*Mi*C';
Hn=Hn/max(max(Hn));
Hns1{i}=Hn;
errn1(i)=norm(H-Hn,'fro')/norm(H,'fro');
ranksn1(i)=rank(Hn);



%%%%
Q = Hn;
Hs = (inv(Q+eye(d)*rhos(i)*10))*(Q*Q');%+rhos(i)*eye(d));
%Hs=B'*S'*S*B/(n*mx);
Hs=Hs/max(max(Hs));
Hss1{i}=Hs;
errs1(i)=norm(H-Hs,'fro')/norm(H,'fro');
ranks1(i)=rank(Hs);

%%%%%

end



plot(log10(rhos),errs,'--o','LineWidth',2,'DisplayName','NS eff. dim');hold on;
plot(log10(rhos),errs1,'--p','LineWidth',2,'DisplayName','NS 10');hold on;
plot(log10(rhos),errn,'-s','LineWidth',2,'DisplayName','Nys eff. dim');hold on;
plot(log10(rhos),errn1,'-<','LineWidth',2,'DisplayName','Nys 10');hold on;
xticks([-5 -4 -3 -2 -1 0]);
xticklabels({'1e-5','1e-4','1e-3','1e-2','1e-1','1e-0'});
ylabel('Relative error');
xlabel('Regularizer log-scale');
ax=gca;
ax.FontSize=24;

figure;
plot(log10(rhos),ranks,'--o','LineWidth',2,'DisplayName','NS eff. dim');hold on;
plot(log10(rhos),ranks1,'--p','LineWidth',2,'DisplayName','NS 10');hold on;
plot(log10(rhos),ranksn,'-s','LineWidth',2,'DisplayName','Nys eff. dim');hold on;
plot(log10(rhos),ranksn1,'-<','LineWidth',2,'DisplayName','Nys 10');hold on;
xticks([-5 -4 -3 -2 -1 0]);
xticklabels({'1e-5','1e-4','1e-3','1e-2','1e-1','1e-0'});
ay=gca;
ay.FontSize=24;
ylabel('Approximation rank');
xlabel('Regularizer log-scale');

figure;
ix = 1;
subplot(5,2,1);imagesc(H);title('Computed Hessian');ylabel('Original');
subplot(5,2,2);imagesc(H-H);title('Difference')
%[va,ix]=min(errs1);
subplot(5,2,3);imagesc(Hss{ix});ylabel('Nys-LM 10 dim')
subplot(5,2,4);imagesc(Hss{ix}-H);%title(' Newton-Sketch with eff. dim.')
%[va,ix]=min(errn1);
subplot(5,2,5);imagesc(Hns{ix});ylabel('Nys 10 dim')
subplot(5,2,6);imagesc(Hns{ix}-H);%title('(d) Newton Sketch with 10 times the dim.')
%[va,ix]=min(errn);
subplot(5,2,7);imagesc(Hss1{ix});ylabel('Nys-LM 20 dim')
subplot(5,2,8);imagesc(Hss1{ix}-H);%title('(b) Nys-Hessian with 10 Columns')
%[va,ix]=min(errn);
subplot(5,2,9);imagesc(Hns1{ix});ylabel('Nys 20 dim')
subplot(5,2,10);imagesc(Hns1{ix}-H);%title('(b) Nys-Hessian with 10 Columns')
set(findobj(gcf,'type','axes'),'FontSize',14);


