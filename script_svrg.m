function [] = main(dataset,dropout,lambda_factor,seed,nepochs,init_epochs,setting,loss)

if isdeployed
   dataset=str2num(dataset);  % see below for the choice of parameter
   dropout=str2num(dropout);  % dropout paramter
   lambda_factor=str2num(lambda_factor); % default parameter 10 => lambda= 1/10n
   seed=str2num(seed); % change the random seed for the experiments
   nepochs=str2num(nepochs); % number of epochs required
   setting=str2num(setting); % see below for the algorithm used
   loss=str2num(loss);  % 0 is logistic, 1 is squared hinge loss
end

name=sprintf('exps/exp_data%d_s%d_d%d_l%d_n%d_%d_seed%d_loss%d.mat',dataset,setting,dropout,lambda_factor,nepochs,init_epochs,seed,loss);
name

if dataset==1
%%%%% Dataset 1 - CKN %%%%%%
   load('/path_data/ckn_matrix.mat');
   X=psiTr;
   n=size(X,2);
   y=-ones(n,1);
   y(find(Ytr==0))=1;
elseif dataset==2
   %%%%% Dataset 2 - gene %%%%%%
   load('/path_data/vant.mat');
   X=X';
   mex_normalize(X);
   y=Y(:,2);
elseif dataset==3
   load('/path_data/alpha.full_norm.mat');
   y=y(1:250000);
   X=X(:,1:250000);
   mex_normalize(X);
end
X=double(X);
y=double(y);
n=size(X,2);

param.lambda=1/(lambda_factor*n);  %% This is the regularization parameter
param.seed=seed;
param.loss=loss;
param.epochs=nepochs;
param.threads=1;
param.dropout=dropout;
param.minibatch=1;
if (dropout==0)
   param.eval_freq=1;
else
   param.eval_freq=5;
end
w0=zeros(size(X,1),1);
if loss==0
   L=0.25;
else 
   L=1;
end
   
% default parameters
param.sgd=false;
param.averaging=false; 
param.accelerated=false;
param.decreasing=false; % decreasing learning rate
param.L=L;

if setting==1
   %%%% Exp 1 - SVRG with 1/12L
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting==2
   %%%% Exp 2 - SVRG with 1/3L (not allowed by theory)
   param.L=L/4;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting==3
   %%%% Exp 3 - accelerated SVRG - constant step size
   param.accelerated=true;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting==4
   %%%% Exp 4 - decreasing - SVRG based on 1/12L
   param.decreasing=true;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting==5
   %%%% Exp 5 - decreasing - SVRG based on 1/3L
   param.L=L/4;
   param.decreasing=true;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting==6
   %%%% Exp 5 - decreasing - acc SVRG based on 1/3L
   param.accelerated=true;
   param.decreasing=true;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting == 7
   %%%% Exp 1 - SGD with 1/L
   param.sgd=true;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting == 8
   %%%% Exp 1 - acc SGD with 1/L
   param.sgd=true;
   param.accelerated=true;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting == 9
   %%%% Exp 1 - SGD decreasing with 1/L
   param.sgd=true;
   param.decreasing=true;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting == 10
   %%%% Exp 1 - acc SGD decreasing with 1/L
   param.accelerated=true;
   param.sgd=true;
   param.decreasing=true;
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
elseif setting == 11
   %%%% Exp 1 - mb-acc-SGD decreasing with 1/L
   param.accelerated=true;
   param.sgd=true;
   param.decreasing=true;
   param.minibatch=round(sqrt(param.L/param.lambda));
   [w logs_exp]=mex_svm_svrg(y,X,w0,param);
end

save(name,'logs_exp','w');

