path_mkl='/path_intel/compilers_and_libraries/linux/mkl/lib/intel64/';
include_mkl='/path_intel/mkl/include/';
pathlibiomp='/path_intel/compilers_and_libraries_2017/linux/lib/intel64/';
path_icc='/path_intel/compilers_and_libraries/linux/';
path_matlab='/path_matlab/matlab-2016a/bin/';
path_libstd='/usr/lib/gcc/x86_64-linux-gnu/5/'; 

%%%%%% list of mex files %%%%
names_mklst={'mex_svm_svrg','mex_normalize'};

%%%%%% various flags %%%%%
format compact;
compiler_icc=[path_icc '/bin/intel64/icpc'];
lib_mkl_sequential=sprintf('-Wl,--start-group %slibmkl_intel_ilp64.a %slibmkl_sequential.a %slibmkl_core.a -Wl,--end-group',path_mkl,path_mkl,path_mkl);
lib_mkl_mt=sprintf('-Wl,--start-group %slibmkl_intel_ilp64.a %slibmkl_intel_thread.a %slibmkl_core.a -Wl,--end-group -L%s -liomp5 -ldl',path_mkl,path_mkl,path_mkl,pathlibiomp);
lib_openmp='-liomp5';
defines='-DNDEBUG -DHAVE_MKL -DINT_64BITS -DAXPBY';
flags='-O3 -fopenmp -static-intel -fno-alias -align -falign-functions';
lflags='';
includes=sprintf('-I./utils/ -I%s',include_mkl);

fid=fopen('run_matlab.sh','w+');
fprintf(fid,'#!/bin/sh\n');
fprintf(fid,sprintf('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%s:%s:%s\n',[path_icc 'lib/intel64/'],path_mkl,[path_cuda 'lib64/']));
fprintf(fid,sprintf('export LIB_INTEL=%s\n',[path_icc 'lib/intel64/']));
fprintf(fid,'export KMP_AFFINITY=verbose,granularity=fine,compact,1,0\n');fprintf(fid,'export LD_PRELOAD=$LIB_INTEL/libimf.so:$LIB_INTEL/libintlc.so.5:$LIB_INTEL/libiomp5.so:$LIB_INTEL/libsvml.so:%s/libstdc++.so\n',path_libstd);
fprintf(fid,[path_matlab 'matlab -nodisplay -singleCompThread -r \"addpath(''./mex/''); "\n']); 
fclose(fid);
!chmod +x run_matlab.sh

for ii=1:length(names_mklst)
   name=names_mklst{ii};
   name
   str=sprintf(' -v -largeArrayDims CXX="%s" DEFINES="\\$DEFINES %s" CXXFLAGS="\\$CXXFLAGS %s" LDFLAGS="\\$LDFLAGS " INCLUDE="\\$INCLUDE %s" LINKLIBS="\\$LINKLIBS -L"%s" %s %s" mex/%s.cpp -output mex/%s.mexa64',compiler_icc,defines,flags,includes,path_mkl,lib_mkl_mt,lib_openmp,name,name);
   args = regexp(str, '\s+', 'split');
   args = args(find(~cellfun(@isempty, args)));
   mex(args{:});
end
