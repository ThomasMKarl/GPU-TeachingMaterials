#build module
#-c: compile only
#--compiler-options '-fPIC' (g++ -fPIC): generates position-independent code, can be included anywhere in the code
nvcc -c module.cu --compiler-options '-fPIC'

#conver to static library with archiver
#c: creates a new archive
#r: replaces old archive
#s: adds object file to archive
ar crs libmodule.a module.o
#create an index for the functions
ranlib libmodule.a

#compile into shared libary (module is cached and loaded)
#name should be lib<name>.so.<major_version>.<minor_version>.<release_number>
nvcc -shared -o libmodule.so module.o

#.o file not needed
rm -f module.o

#copy library (make install) in system path (e.g. /usr/local/lib) or modify environment variable LD_LIBRARY_PATH
#file should be copy or symlink to relese version that is needed
#e.g.: sudo cp ./libmodule.so.1.0 /usr/local/lib/ | sudo ln -s /usr/local/lib/libmodule.so.1.0 /usr/local/lib/libmodule.so
#run ldconfig
#tell linker to look for libmodule.so or libmodule.a in search path via -l
g++ main.cpp -lcudart -lmodule


#dynamic case: user must provide library to execute binary, library is loaded in cache (usuallay faster, smaller binary)
#static case: library is part of the binary (no dependencies, but slower)
