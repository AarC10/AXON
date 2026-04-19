add_library(nn
    Module.cpp
    Linear.cpp
    Serialization.cpp
    activations/ReLU.cpp
    activations/Sigmoid.cpp
)

target_link_libraries(nn PUBLIC core)

if(AXON_ENABLE_OPENMP)
  target_link_libraries(nn PUBLIC OpenMP::OpenMP_CXX)
endif()
