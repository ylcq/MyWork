#ifndef __PALLOGRMODELTRAINER_H
#define __PALLOGRMODELTRAINER_H

#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/Model.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataStruct.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataLoader.h"

namespace AFL_PAL
{
namespace LOGR
{
class LogRModelTrainer
{
public:
    struct Args
    {
        LR_ARGUMENTS_NEW& LR_Args;
        TRexCommonObjects::InternalTable* arhsTab;
    };

public:
    explicit LogRModelTrainer(ltt::allocator& alloc):
        m_alloc(&alloc)
    { }

public:
    _STL::unique_ptr<LogRModel> operator()(
                    LogRTrainData const* pData,
                    TRexCommonObjects::InternalTable const* paramTBL,
                    Args const* pArgs,
                    size_t threadNum,
                    double* pErr = nullptr,
                    _STL::vector<double>* pErrlog = nullptr) const;

private:
    ltt::allocator* m_alloc;
};

}// end of LOGR
}// end of AFL_PAL

#endif

