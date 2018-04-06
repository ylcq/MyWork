#ifndef __PALLOGRHELPER_H
#define __PALLOGRHELPER_H

#include "AFL/PAL/pal.h"
#include "TRexCommonObjects/InternalTable/InternalTable.h"
#include "AFL/PAL/APIExport.hpp"
#include "AFL/PAL/Utilities/CrossValidation/ParameterGroup.h"
#include "AFL/PAL/Utilities/CrossValidation/ParameterSelector.h"
#include "AFL/PAL/Utilities/CrossValidation/ParameterSearch.h"
#include "AFL/PAL/Utilities/CrossValidation/Help.h"

namespace AFL_PAL
{
namespace LOGR
{
enum optMethod
{
    NEWTON = 0,
    GRADIENT = 1,
    CCD = 2,
    LBFGS = 3,
    SGD = 4,
    BFGS = 5,
    PGD = 6
};

enum metricType
{
    ACCURACY = 0,
    F1_SCORE = 1,
    AUC = 2,
    NLL = 3
};

// used parameters for logistic regression
struct LR_ARGUMENTS_NEW
{
    int32_t         betaNumber;
    int32_t         maxIteration;
    int32_t         threadNumber;
    int32_t         pmmlFlag;
    int32_t         method;
    int32_t         pvalue;
    int32_t         sgdType;
    int32_t         learningRateType;
    int32_t         seed;
    int32_t         forcePass;
    int32_t         passNum;
    int32_t         hasId;
    int32_t         standardize;
    double          alpha;
    double          lambda;
    double          threshold;
    double          epsilon;
    double          sgdAlpha;
    double          sgdGamma;
    double          sgdC;
    double          lbfgsM;
    double          c;

    bool            useGridSearch;
    int             random_search_times;
    int             foldNum;
    int             trainForm;
    uint64_t        timeout;
    bool            pIsTimeout;
    bool            pHasRsl;
    int             metric_type;
    CV::CvParameters cvPrms;
    
    ltt_adp::string featureStr;
    ltt_adp::string depVar;

    explicit LR_ARGUMENTS_NEW(ltt::allocator& alloc):
        featureStr(alloc),
        depVar(alloc)
    {
    }
};



}// end of LOGR
}// end of AFL_PAL
#endif



