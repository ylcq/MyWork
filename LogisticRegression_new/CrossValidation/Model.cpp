#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/Model.h"
#include <chrono>

USE_TRACE(TRACE_PAL_LOGISTICREGRESSION);
USE_ERROR(AFL_PAL, ERR_PAL_COMM_NOT_CONVERGENCE);
USE_TRACE(TRACE_PAL_TEST);

namespace AFL_PAL
{
namespace LOGR
{
void LogRModel::modelTrain(dense_sparse_matrix::DSMatrix& dsMatrix, Eigen::VectorXd& yValue,
                LR_ARGUMENTS_NEW& LR_Args, TRexCommonObjects::InternalTable* argsTab,
                double lambda, double alpha, size_t threadNumber)
{


}


}// end of LOGR
}// end of AFL_PAL


