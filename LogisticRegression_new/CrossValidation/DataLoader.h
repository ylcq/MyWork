#ifndef __PALLOGRDATALOADER_H
#define __PALLOGRDATALOADER_H

#include "ltt/map.hpp"
#include "ltt/string.hpp"
#include "AFL/PAL/pal.h"
#include "AFL/PAL/palParameters.h"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/DataTableView/DataTableView.h"
#include "AFL/PAL/Utilities/CrossValidation/ParameterGroup.h"
#include "AFL/PAL/Utilities/CrossValidation/ParameterSelector.h"
#include "AFL/PAL/Utilities/CrossValidation/ParameterSearch.h"
#include "AFL/PAL/Classification/LogisticRegression_new/PALLogisticRegression_fix.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataStruct.h"

namespace AFL_PAL
{
namespace LOGR
{
class LogRDataLoader
{
public:
     struct Args
     {
         LR_ARGUMENTS_NEW& LR_Args;
         ltt::vector<_STL::string>& classMap;
         ltt_adp::vector<bool>& isCategoryVec;
         bool& isCV;
     };

public:
    explicit LogRDataLoader(ltt::allocator& alloc):
        m_alloc(&alloc)
    {  }

public:
    _STL::unique_ptr<LogRTrainData> loadTrainData(
                 Utility::DataTableViewBase const* pDataView,
                 Args const* pArgs,
                 size_t threadNumber) const;

    _STL::unique_ptr<LogRPredictData> loadPredictData(
                 Utility::DataTableViewBase const* pDataView,
                 Args const* pArgs,
                 size_t threadNumber) const;

    _STL::unique_ptr<LogRCompareData> loadCompareData(
                 Utility::DataTableViewBase const* pDataView,
                 Args const* pArgs,
                 size_t threadNumber) const;

private:
    ltt::allocator* m_alloc;

};


}// end of LOGR
}// end of AFL_PAL


#endif










