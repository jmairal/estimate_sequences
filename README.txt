The package contains the code used for all the experiments of the ICML submission
``Estimate Sequences for Variance-Reduced Stochastic Composite Optimization''

* The algorithms are entirely coded in C++ and are available in the file svrg.h

* The C++ code is interfaced with Matlab and the script script_svrg.m loads the
  datasets (not provided here because of size limitation), and run the
  experiments, depending on the setting chosen.

* The C++ code is compiled and turned into mex files by the script build.m, which requires
the Intel C++ compiler and the MKL library (version 2017 was used), as well as matlab 
installation (version 2016a was used). The script build.m creates another
script called "run_matlab.sh", which need to be launched in a terminal to run matlab
and be able to run the mex files. We use ubuntu 16.04 to compile and run all
experiments. In summary to use our code
 1) edit build.m and setup the right paths to the intel compilers, matlab, and libstdc++ libraries.
 2) open matlab and type "build"
 3) close matlab
 4) open matlab in a terminal using "sh run_matlab.sh"
 5) you may now use script_svrg to run some experiments. You may need to edit
 the script to load the dataset of your choice, which should contain a matrix X
 of size p times n and a binary vector y of size n with -1 or +1 entries.
 
* The folder utils is of no concern for this ICML submission. It simply
  contains a Matrix library coming from a different open-source software.
