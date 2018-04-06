#ifndef __PALLOGRMODELEVALUATOR_H
#define __PALLOGRMODELEVALUATOR_H

#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/Model.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataStruct.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataLoader.h"

namespace AFL_PAL
{
namespace LOGR
{
class LogREvaluator
{
public:
    struct Args
    {
        LR_ARGUMENTS_NEW& LR_Args;
        ltt::vector<_STL::string>& classMap;
    };

public:
    explicit LogREvaluator(ltt::allocator& alloc):
        m_alloc(&alloc)
    {  }

    double operator()(
        LogRModel const* pModel,
        LogRPredictData const* pPdData,
        LogRCompareData const* pCmData,
        Args const* pArgs,
        size_t threadNumber) const;


private:
    ltt::allocator* m_alloc;

};


}// end of LOGR
}// end of AFL_PAL

#endif
