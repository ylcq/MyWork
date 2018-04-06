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
    int method = LR_Args.method;

    uint32_t nrow = dsMatrix.rows();
    uint32_t ncol = dsMatrix.cols();
    expandColSize = ncol-1;
    feature_dimension = ncol;

    m_coefficient.resize(feature_dimension);

    if(method == SGD){ // sgd optimization

    }
    else if(method == CCD){// ccd optimization

    }
    else if(method == PGD){// pgd optimization
         ENetRegularizer_new::lrPGDSolver(
             dsMatrix,
             yValue,
             lambda,
             alpha,
             LR_Args.maxIteration,
             threadNumber,
             finalIter,
             m_coefficient,
             LR_Args.threshold,
             m_alloc);
    }
    else{
        OptimizationMethod om = OptimizationMethod::NEWTON;
        LineSearchStrategy lss = LineSearchStrategy::FIXEDSTEP;
        ConvergenceStrategy cs = ConvergenceStrategy::CONVLR;
        OptimParam op;
        op.step = 1.0;
        uint32_t mydim = expandColSize + 1;

        ObjectFunction myFun(dsMatrix, yValue, mydim, threadNumber, *m_alloc, lambda);

        switch(method){
            // newton
            case NEWTON:
            default:
            {
                om = OptimizationMethod::NEWTON;
                lss = LineSearchStrategy::ARMIJO;
                cs = ConvergenceStrategy::CONVLR;
                op.xtol = LR_Args.epsilon;
                break;
            }
            case LBFGS:
            {
                om = OptimizationMethod::LBFGS;
                lss = LineSearchStrategy::WOLFE;
                cs = ConvergenceStrategy::CONVSTRICT;
                op.m = LR_Args.lbfgsM;
                op.epsilon = LR_Args.epsilon;
                break;
            }
        }
        op.maxIter = LR_Args.maxIteration;
        op.ftol = LR_Args.threshold;
        Solver<ObjectFunction> mySolver(*m_alloc);

        mySolver.init(m_coefficient, op, myFun, om, lss, cs);
        const int ret = mySolver.solve();
        solutionStatus = ret;
        if(ret == -1)
        {
            TRACE_ERROR(TRACE_PAL_LOGISTICREGRESSION, "solver init error" << ltt::endl);
            THROW_PAL_ERROR(ERR_PAL_COMM_NOT_CONVERGENCE, ltt::msgarg_text("APPEND", "init error"));
        }
        else
        {
            ltt::swap(m_coefficient, mySolver.getX());
        }
        minusMLE = mySolver.getObj();
        finalIter = mySolver.getIter();

    }
}


}// end of LOGR
}// end of AFL_PAL









