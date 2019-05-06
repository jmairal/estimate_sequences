#include <mex.h>
#include <mexutils.h>
#include <svrg.h>

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[],
      const int nlhs) {
   if (!mexCheckType<T>(prhs[0])) 
      mexErrMsgTxt("type of argument 1 is not consistent");
   if (mxIsSparse(prhs[0])) 
      mexErrMsgTxt("argument 1 should not be sparse");

   if (!mexCheckType<T>(prhs[1])) 
      mexErrMsgTxt("type of argument 2 is not consistent");

   if (!mxIsStruct(prhs[3])) 
      mexErrMsgTxt("argument 4 should be a struct");

   T* pry = reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dimsy=mxGetDimensions(prhs[0]);
   INTM my=static_cast<INTM>(dimsy[0]);
   INTM ny=static_cast<INTM>(dimsy[1]);
   Vector<T> y(pry,my*ny);

   T* prX = reinterpret_cast<T*>(mxGetPr(prhs[1]));
   const mwSize* dimsX=mxGetDimensions(prhs[1]);
   INTM p=static_cast<INTM>(dimsX[0]);
   INTM n=static_cast<INTM>(dimsX[1]);
   Matrix<T> X(prX,p,n);

   T* prw0 = reinterpret_cast<T*>(mxGetPr(prhs[2]));
   const mwSize* dimsw0=mxGetDimensions(prhs[2]);
   INTM pw0=static_cast<INTM>(dimsw0[0]);
   INTM nw0=static_cast<INTM>(dimsw0[1]);
   Vector<T> w0(prw0,pw0*nw0);


   plhs[0]=createMatrix<T>(p,1);
   T* prw=reinterpret_cast<T*>(mxGetPr(plhs[0]));
   Vector<T> w(prw,p);
   w.copy(w0);

   srandom(0);
   const int epochs = getScalarStructDef<int>(prhs[3],"epochs",100);
   plhs[1]=createMatrix<T>(epochs,1);
   T* prlogs=reinterpret_cast<T*>(mxGetPr(plhs[1]));
   Vector<T> logs(prlogs,epochs);

   int threads = getScalarStructDef<int>(prhs[3],"threads",-1);
   const T lambda = getScalarStruct<T>(prhs[3],"lambda");
   const bool averaging = getScalarStructDef<bool>(prhs[3],"averaging",false);
   const bool decreasing = getScalarStructDef<bool>(prhs[3],"decreasing",false);
   const bool use_sgd = getScalarStructDef<bool>(prhs[3],"sgd",false);
   const T L = getScalarStruct<T>(prhs[3],"L");
   const T dropout = getScalarStructDef<T>(prhs[3],"dropout",0);
   const int seed = getScalarStruct<int>(prhs[3],"seed");
   const int loss = getScalarStructDef<int>(prhs[3],"loss",1);
   const int eval_freq= getScalarStructDef<int>(prhs[3],"eval_freq",1);
   const int minibatch= getScalarStructDef<int>(prhs[3],"minibatch",1);
   srandom(seed);
   const bool accelerated = getScalarStructDef<T>(prhs[3],"accelerated",false);
   if (threads == -1) {
      threads=1;
#ifdef _OPENMP
      threads =  MIN(MAX_THREADS,omp_get_num_procs());
#endif
   } 
   threads=init_omp(threads);
   if (accelerated) {
      if (use_sgd) {
         acc_sgd(y,X,w,L,lambda,epochs,decreasing,dropout,eval_freq,logs,minibatch,loss);
      } else {
         acc_random_svrg(y,X,w,L,lambda,epochs,decreasing,dropout,eval_freq,logs,loss);
      }
   } else {
      if (use_sgd) {
         sgd(y,X,w,L,lambda,epochs,averaging,decreasing,dropout,eval_freq,logs,loss);
      } else {
         random_svrg(y,X,w,L,lambda,epochs,averaging,decreasing,dropout,eval_freq,logs,loss);
      }
   } 
}

   void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
      if (nrhs != 4)
         mexErrMsgTxt("Bad number of inputs arguments");

      if (nlhs != 2) 
         mexErrMsgTxt("Bad number of output arguments");

      if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
         callFunction<double>(plhs,prhs,nlhs);
      } else {
         callFunction<float>(plhs,prhs,nlhs);
      }
   }




