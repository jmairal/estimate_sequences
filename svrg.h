#ifndef SVM_H
#define SVM_H  

#include "linalg.h"

template <typename T>
void dropout_vec(Vector<T>& tmp, const T dropout) {
   const int n = tmp.n();
   if (dropout)
      for (int ii=0; ii<n; ++ii)
         if (random() <= RAND_MAX*dropout) tmp[ii]=0;
}

template <typename T>
T compute_loss(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq, const int loss) {
   if (loss==0) {
      return compute_loss_logistic(y,X,w,lambda,dropout,freq);
   } else {
      return compute_loss_sqhinge(y,X,w,lambda,dropout,freq);
   }
}

template <typename T>
T compute_loss_logistic(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq) {
   const int n = y.n();
   T loss=0;
   Vector<T> tmp;
   if (dropout) {
      for (int ll=0; ll<freq; ++ll) {
         for (int kk=0; kk<n; ++kk) {
            X.copyCol(kk,tmp);
            dropout_vec(tmp,dropout);
            loss += logexp(-y[kk]*tmp.dot(w));
         }
      }
      loss *= T(1.0)/(freq*n);
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         loss += logexp(-y[kk]*tmp[kk]);
      }
      loss *= T(1.0)/(n);
   } 
   loss += T(0.5)*lambda*w.nrm2sq();
   return loss;
}

template <typename T>
T compute_loss_sqhinge(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, const T lambda, const T dropout, const int freq) {
   const int n = y.n();
   T loss=0;
   Vector<T> tmp;
   if (dropout) {
      for (int ll=0; ll<freq; ++ll) {
         for (int kk=0; kk<n; ++kk) {
            X.copyCol(kk,tmp);
            dropout_vec(tmp,dropout);
            const T los=MAX(0,1-y[kk]*tmp.dot(w));
            loss += los*los;
         }
      }
      loss *= T(0.5)/(freq*n);
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         const T los=MAX(0,1-y[kk]*tmp[kk]);
         loss += los*los;
      }
      loss *= T(0.5)/(n);
   } 
   loss += T(0.5)*lambda*w.nrm2sq();
   return loss;
}

template <typename T>
void compute_grad(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout, const int loss) {
   if (loss==0) {
      return compute_grad_logistic(y,x,w,grad,lambda,dropout);
   } else {
      return compute_grad_sqhinge(y,x,w,grad,lambda,dropout);
   }
}

template <typename T>
void compute_grad_logistic(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   grad.copy(x);
   dropout_vec(grad,dropout);
   const T s = T(1.0)/(T(1.0)+exp_alt<T>(y*grad.dot(w)));
   grad.scal(-y*s);
   grad.add(w,lambda);
}


template <typename T>
void compute_grad_sqhinge(const T y, const Vector<T>& x, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   grad.copy(x);
   dropout_vec(grad,dropout);
   const T s = MAX(0,1-y*grad.dot(w));
   grad.scal(-y*s);
   grad.add(w,lambda);
}

template <typename T>
void compute_fullgrad(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout, const int loss) {
   if (loss==0) {
      return compute_fullgrad_logistic(y,X,w,grad,lambda,dropout);
   } else {
      return compute_fullgrad_sqhinge(y,X,w,grad,lambda,dropout);
   }
}

template <typename T>
void compute_fullgrad_logistic(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   const int n = y.n();
   Vector<T> tmp;
   if (dropout) {
      grad.setZeros();
      grad.resize(w.n());
      for (int kk=0; kk<n; ++kk) {
         X.copyCol(kk,tmp);
         dropout_vec(tmp,dropout);
         const T s = T(1.0)/(T(1.0)+exp_alt<T>(y[kk]*tmp.dot(w)));
         grad.add(tmp,-y[kk]*s);
      }
      grad.scal(T(1.0/n));
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         const T s = T(1.0)/(T(1.0)+exp_alt<T>(y[kk]*tmp[kk]));
         tmp[kk]=-y[kk]*s;
      }
      X.mult(tmp,grad,T(1.0)/n);
   }
   grad.add(w,lambda);
}


template <typename T>
void compute_fullgrad_sqhinge(const Vector<T>& y, const Matrix<T>& X, const Vector<T>& w, Vector<T>& grad, const T lambda, const T dropout) {
   const int n = y.n();
   Vector<T> tmp;
   if (dropout) {
      grad.setZeros();
      grad.resize(w.n());
      for (int kk=0; kk<n; ++kk) {
         X.copyCol(kk,tmp);
         dropout_vec(tmp,dropout);
         const T s=MAX(0,1-y[kk]*tmp.dot(w));
         grad.add(tmp,-y[kk]*s);
      }
      grad.scal(T(1.0/n));
   } else {
      X.multTrans(w,tmp);
      for (int kk=0; kk<n; ++kk) {
         tmp[kk]=-y[kk]*MAX(0,1-y[kk]*tmp[kk]);
      }
      X.mult(tmp,grad,T(1.0)/n);
   }
   grad.add(w,lambda);
}

template <typename T>
void random_svrg(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int epochs, const bool averaging, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, const int param_loss = 1) {
   const int n = y.n();
   const int p = X.m();
   const T eta= T(1.0)/(12*L);
   logs.resize(epochs);
   Vector<T> wav;
   wav.copy(w);
   Vector<T> anchor;
   anchor.copy(w);
   Vector<T> grad_anchor, grad, grad2, col;
   compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss);

   cout << "SVRG" << endl;
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = averaging ? compute_loss(y,X,wav,lambda,dropout,eval_freq,param_loss) : compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss);
         const T etak = decreasing ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
         cout << "Iteration " << ii << " - eta: " << etak << " - obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda,dropout,param_loss);
      compute_grad(y[ind],col,anchor,grad2,lambda,dropout,param_loss);
      grad.add(grad2,-T(1.0));
      grad.add(grad_anchor);

      const T etak = decreasing ? MIN(MIN(eta,T(1.0)/(5*n*lambda)),T(2.0)/(lambda*(ii+2))) : eta;
      w.add(grad,-etak);
      if (random() % n == 0) {
         anchor.copy(w);
         compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss);
      }
      if (averaging) {
         const T tau = MIN(lambda*etak,T(1.0)/(5*n));
         wav.scal((T(1.0)-tau));
         wav.add(w,tau);
      }
   }
   if (averaging)
      w.copy(wav);
}

template <typename T>
void acc_random_svrg(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int epochs, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, const int param_loss = 1) {
   const int n = y.n();
   const int p = X.m();
   const T eta= MIN(T(1.0)/(3*L), T(1.0)/(5*lambda*n));
   logs.resize(epochs);
   Vector<T> anchor, grad_anchor, grad, grad2, col, y_k, v_k;
   anchor.copy(w);
   y_k.copy(w);
   v_k.copy(w);
   compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss);

   cout << "Accelerated SVRG" << endl;
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss);
         const T etak = decreasing ? MIN(eta,12*n/(5*lambda*(T(ii+1)*T(ii+1))))  : eta;
         cout << "Iteration " << ii << " - eta: " << etak << " - obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
      }
      const T etak = decreasing ? MIN(eta,12*n/(5*lambda*(T(ii+1)*T(ii+1))))  : eta;
      const T deltak=sqrt(T(5.0)*etak*lambda/(3*n));
      const T thetak=(3*n*deltak-5*lambda*etak)/(3-5*lambda*etak);
      y_k.copy(v_k);
      y_k.scal(thetak);
      y_k.add(anchor,T(1.0-thetak));

      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,y_k,grad,lambda,dropout,param_loss);
      compute_grad(y[ind],col,anchor,grad2,lambda,dropout,param_loss);
      grad.add(grad2,-T(1.0));
      grad.add(grad_anchor);
      w.copy(y_k);
      w.add(grad,-etak);

      v_k.scal(1-deltak);
      v_k.add(y_k,deltak);
      v_k.add(w,(deltak/(lambda*etak)));
      v_k.add(y_k,-(deltak/(lambda*etak)));

      if (random() % n == 0) {
         anchor.copy(w);
         compute_fullgrad(y,X,anchor,grad_anchor,lambda,dropout,param_loss);
      }
   }
}

template <typename T>
void sgd(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int epochs, const bool averaging, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, const int param_loss = 1) {
   const int n = y.n();
   const int p = X.m();
   const T eta= T(1.0)/(L);
   logs.resize(epochs);
   Vector<T> wav;
   wav.copy(w);
   Vector<T> grad, col;

   cout << "SGD" << endl;
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = averaging ? compute_loss(y,X,wav,lambda,dropout,eval_freq,param_loss) : compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss);
         const T etak = decreasing ? MIN(eta,T(2.0)/(lambda*(ii+2))) : eta;
         cout << "Iteration " << ii << " -- eta " << etak << " -- obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,w,grad,lambda,dropout,param_loss);

      const T etak = decreasing ? MIN(eta,T(2.0)/(lambda*(ii+2))) : eta;
      w.add(grad,-etak);
      if (averaging) {
         const T tau = lambda*etak;
         wav.scal((T(1.0)-tau));
         wav.add(w,tau);
      }
   }
   if (averaging)
      w.copy(wav);
}

template <typename T>
void acc_sgd(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int epochs, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, const int param_loss = 1) {
   const int n = y.n();
   const int p = X.m();
   const T eta= (T(1.0)/(L));
   logs.resize(epochs);
   Vector<T> yk, wold;
   yk.copy(w);
   Vector<T> grad, col;

   cout << "Acc SGD" << endl;
   for (int ii = 0; ii<n*epochs; ++ii) {
      if ((ii % (eval_freq*n)) == 0) {
         const T loss = compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss);
         const T etak = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;
         cout << "Iteration " << ii << " -- eta " << etak  << " -- obj " <<  loss << endl;
         logs[ii/(eval_freq*n)]=loss;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,yk,grad,lambda,dropout,param_loss);

      const T etak = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;

      wold.copy(w);
      w.copy(yk);
      w.add(grad,-etak);
      const T etakp1 = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+3)*T(ii+3))) : eta;
      const T deltak=sqrt(lambda*etak);
      const T deltakp1=sqrt(lambda*etakp1);
      const T betak=deltak*(1-deltak)*etakp1/(etak*deltakp1+ etakp1*deltak*deltak);
      yk.copy(w);
      wold.add(w,-T(1.0));
      yk.add(wold,-betak);
   }
}

template <typename T>
void acc_sgd(const Vector<T>& y, const Matrix<T>& X, Vector<T>& w, const T L, const T lambda, const int epochs, const bool decreasing, const T dropout, const int eval_freq, Vector<T>& logs, const int mb = 1, const int param_loss = 1) {
   const int n = y.n();
   const int p = X.m();
   const T eta= (T(1.0)/(L));
   logs.resize(epochs);
   Vector<T> yk, wold;
   yk.copy(w);
   Vector<T> grad, grad2, col;

   cout << "Acc SGD" << endl;
   const int num_iter= (n*epochs)/mb;
   const int freq_epoch= n/mb;
   int last_log=0;
   for (int ii = 0; ii<num_iter; ++ii) {
      if (ii*mb/(eval_freq*n) >= last_log) {
         const T loss = compute_loss(y,X,w,lambda,dropout,eval_freq,param_loss);
         const T etak = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;
         cout << "Iteration " << ii << " -- eta " << etak  << " -- obj " <<  loss << endl;
         logs[last_log++]=loss;
      }
      const int ind = random() % n;
      X.refCol(ind,col);
      compute_grad(y[ind],col,yk,grad,lambda,dropout,param_loss);
      if (mb > 1) {
         for (int ii=2; ii<mb; ++ii) {
            const int ind = random() % n;
            X.refCol(ind,col);
            compute_grad(y[ind],col,yk,grad2,lambda,dropout,param_loss);
            grad.add(grad2);
         }
         grad.scal(T(1.0)/mb);
      }

      const T etak = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+2)*T(ii+2))) : eta;

      wold.copy(w);
      w.copy(yk);
      w.add(grad,-etak);
      const T etakp1 = decreasing ? MIN(eta,T(4.0)/(lambda*T(ii+3)*T(ii+3))) : eta;
      const T deltak=sqrt(lambda*etak);
      const T deltakp1=sqrt(lambda*etakp1);
      const T betak=deltak*(1-deltak)*etakp1/(etak*deltakp1+ etakp1*deltak*deltak);
      yk.copy(w);
      wold.add(w,-T(1.0));
      yk.add(wold,-betak);
   }
}





#endif
