add_library(nn
    Module.cpp
    Linear.cpp
    Serialization.cpp
    activations/ReLU.cpp
    activations/Sigmoid.cpp
)

target_link_libraries(nn PUBLIC core)
