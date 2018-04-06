#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataLoader.h"

namespace AFL_PAL
{
namespace LOGR
{
_STL::unique_ptr<LogRTrainData> LogRDataLoader::loadTrainData(
         Utility::DataTableViewBase const* pDataView,
         Args const* pArgs,
         size_t threadNumber) const
{
    _STL::unique_ptr<LogRTrainData> trainData(new(*m_alloc) LogRTrainData(*m_alloc));

    int method = pArgs->LR_Args.method;
    trainData->loadData(pDataView, pArgs->classMap, pArgs->isCategoryVec, method, pArgs->isCV);
    if(pArgs->LR_Args.standardize == 1){
        trainData->standardize();
    }

    return trainData;

}

_STL::unique_ptr<LogRPredictData> LogRDataLoader::loadPredictData(
         Utility::DataTableViewBase const* pDataView,
         Args const* pArgs,
         size_t threadNumber) const
{
    _STL::unique_ptr<LogRPredictData> predictData(new(*m_alloc) LogRPredictData(*m_alloc));

    predictData->loadData(pDataView, pArgs->isCategoryVec);

    return predictData;

}

_STL::unique_ptr<LogRCompareData> LogRDataLoader::loadCompareData(
         Utility::DataTableViewBase const* pDataView,
         Args const* pArgs,
         size_t threadNumber) const
{
    _STL::unique_ptr<LogRCompareData> compareData(new(*m_alloc) LogRCompareData(*m_alloc));

    compareData->loadData(pDataView, pArgs->classMap);

    return compareData;

}


}// end of LOGR
}// end of AFL_PAL









