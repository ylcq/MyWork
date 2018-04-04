ADD_SUBDIRECTORY(CrossValidation)
ADD_SUBDIRECTORY(test)

SET(pal_logr_new_src
    CrossValidation/DataLoader.cpp
    CrossValidation/DataStruct.cpp
    CrossValidation/Model.cpp
    CrossValidation/ModelEvaluator.cpp
    CrossValidation/ModelTrainer.cpp
    PALLogisticRegression_fix.cpp
    PALLREnetRegularizer_new.cpp
)

ADD_AFL_LIBRARY(pal_logr_new
    STATIC
        ${pal_logr_new_src}
)

TARGET_COMPILE_FOR_LIBRARIES(pal_logr_new
    external_spine
    external_boost
    external_eigen
)

TARGET_LINK_LIBRARIES(pal_logr_new
    PRIVATE
        hdbsax
        hdbcscommon1
        hdbrskernel
        pal_decisiontree
        pal_preproc
        pal_util
        pal_psme
        pal_state
        pal_basis
)
