cmake_minimum_required(VERSION 2.4)
include_directories( liblbfgs/include )
add_library( lbfgs liblbfgs/lib/lbfgs.c )
