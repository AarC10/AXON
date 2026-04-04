add_library(loss
    MSELoss.cpp
    CrossEntropyLoss.cpp
)

target_link_libraries(loss PUBLIC core)
