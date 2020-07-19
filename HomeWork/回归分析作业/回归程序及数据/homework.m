clc, clear
origin_xdata=xlsread('粮食数据1.xlsx');
origin_xdata=origin_xdata(2:end,2:end);
xdata(:, 1:6) = origin_xdata(:, 2:7)
xdata(:, 7) = origin_xdata(:, 1)
% x=zscore(xdata(:,2:7));%读取自变量数据矩阵
% y1=zscore(xdata(:,1));%读取因变量数据矩阵
x = xdata(:,1:6);
y1 = xdata(:, 7);
reglm(y1,x)   


%逐步回归
inmodel=1:6;     %进入模型的变量为前6个
stepwise(x,y1,inmodel)   %逐步回归

%%%%主成分回归

xz = zscore(x);%数据标准化
[coeff,score,latent,tsquare,explained]=pca(xz)   %由观测数据矩阵作分析
z1=score(:,[1:6]);
reglm(y1,z1)%发现第五个主成分得分量对因变量影响不显著（p>0.05）,因而删除它！
z1=score(:,[1 2 3 4 6]);
reglm(y1,z1)

%若只考虑前三个主成分，则拟合优度大大降低
z1=score(:,[1:3]);
reglm(y1,z1)

%计算因变量对原始自变量的回归方程系数
xn=zscore(x);
yn=zscore(y1);
d=xn*coeff;
st=coeff(:,[1 2 3 4 6])*(d(:,[1 2 3 4 6])\yn);
st2=[mean(y1)-std(y1)*mean(x)./std(x)*st,std(y1)*st'./std(x)],

%直接考虑前三个主成分时的回归方程式
st=coeff(:,[1:3])*(d(:,[1:3])\yn);
st3=[mean(y1)-std(y1)*mean(x)./std(x)*st,std(y1)*st'./std(x)],


%只考虑一个因变量时
mu=mean(xdata(:,1:7));sig=std(xdata(:,1:7)); %求均值和标准差
ab=zscore(xdata(:,1:7)); %数据标准化
a=ab(:,1:6);b1=ab(:,7);
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(a,b1)%观测整体所有成分对的情况
ncomp=6; %根据整体情况，选择成分的对数
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(a,b1,ncomp)
contr=cumsum(PCTVAR,2) %求累积贡献率
n=size(a,2); m=size(b1,2); %n是自变量的个数,m是因变量的个数
BETA2(1,:)=mu(n+1:end)-mu(1:n)./sig(1:n)*BETA([2:end],:).*sig(n+1:end); %原始数据回归方程的常数项
BETA2([2:n+1],:)=(1./sig(1:n))'*sig(n+1:end).*BETA([2:end],:) %计算原始自变量x1,...,xn的系数，每一列是一个回归方程
