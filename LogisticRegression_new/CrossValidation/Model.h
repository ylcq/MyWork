#ifndef __PALLOGRMODEL_H
#define __PALLOGRMODEL_H

#include <ltt/adp/vector.hpp>
#include <TRexUtils/Parallel.h>
#include <TRexUtils/ParallelDispatcher.h>
#include "AFL/PAL/pal.h"
#include "AFL/PAL/Classification/LogisticRegression_new/PALLogisticRegression_fix.h"
#include "AFL/PAL/Classification/LogisticRegression_new/PALLREnetRegularizer_new.h"
#include "AFL/PAL/PALTableTypeVerifier.h"
#include "AFL/PAL/Statistics/CommonFunctions/PValueCalc.hpp"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/Matrix/DenseSparseMatrix.hpp"
#include "AFL/PAL/Utilities/Optimization/Solver.hpp"
#include "AFL/PAL/Utilities/PMML/palGenPmml.h"
#include "AFL/PAL/Utilities/PMML/regressionModel.h"
#include "AFL/PAL/palCancel.h"
#include "AFL/PAL/palCommonFunction.hpp"
#include "AFL/PAL/palParameters.h"
#include "AFL/PAL/palTableTypes.h"

namespace AFL_PAL
{
namespace LOGR
{
class LogRModel
{
public:
    explicit LogRModel(ltt::allocator& alloc):
        m_alloc(&alloc)
    {  }

public:
    void modelTrain(dense_sparse_matrix::DSMatrix& dsMatrix, Eigen::VectorXd& yValue,
                    LR_ARGUMENTS_NEW& LR_Args, TRexCommonObjects::InternalTable* argsTab,
                    double lambda, double alpha, size_t threadNumber);

private:
    ltt::allocator* m_alloc;

};


}// end of LOGR
}// end of AFL_PAL

#endif
