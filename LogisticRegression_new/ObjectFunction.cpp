/*
 * ObjectFunction.cpp
 *
 *  Created on: Dec 27, 2013
 *      Author: root
 */
#include "AFL/PAL/Classification/LogisticRegression/ObjectFunction.h"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/Matrix/Utilities.hpp"
#include "AFL/PAL/Classification/LogisticRegression/MathCalculation.hpp"

//USE_TRACE(TRACE_PAL_MATRIX);
//USE_ERROR(AFL_PAL, ERR_PAL_COMM_MATRIX_INTERNAL_ERR);

namespace AFL_PAL
{
namespace LOGR
{
// obj = logistic likelihood
// only update object function value
int ObjectFunction::updateObjective(struct State& out_state)
{
    EigenMapVec xVec(out_state.x.data(), m_Dimension);
    EigenMapVec thetaVec(m_thetax.data(), m_ttl);
    dense_sparse_matrix::matrixVectorProduct<EigenMapVec, EigenMapVec>(m_vecData, xVec, thetaVec, m_numthread, m_Alloc);

    double sum = 0;

    EigenMapArr thetaxArr(m_thetax.data(), m_ttl);
    EigenMapArr catArr(m_vecCat.data(), m_ttl);
    EigenMapArr pArr(m_p.data(), m_ttl);
    calculateLogExpByParallel(thetaxArr, catArr, m_numthread, m_Alloc, sum, pArr);

    sum /= m_ttl;
    double L2 = 0.0;
    for(size_t i=1; i<m_Dimension; ++i)
    {
        L2 += out_state.x[i] * out_state.x[i];
    }
    L2 = 0.5 * m_c * L2;
    sum += L2;

    out_state.obj = sum;

    return 0;
}

int ObjectFunction::updateGradient(struct State& out_state)
{
    EigenMapVec pVec(m_p.data(), m_ttl);
    EigenMapVec catVec(m_vecCat.data(), m_ttl);
    EigenMapVec pyVec(m_py.data(), m_ttl);
    pyVec = pVec - catVec;

    EigenVec grad(m_Dimension);
    EigenMapVec gradVec(out_state.grad.data(), m_Dimension);
    dense_sparse_matrix::matrixVectorProductAtV<EigenMapVec, EigenVec>(dataMat, pyVec, grad, m_Alloc, m_numthread);
    gradVec = grad;

    for(size_t i=0; i<m_Dimension; ++i)
    {
        out_state.grad[i] /= m_ttl;
    }
    for(size_t i=1; i<m_Dimension; ++i)
    {
        out_state.grad[i] += m_c * out_state.x[i];
    }

    return 0;
}

int ObjectFunction::updateState(struct State& out_state)
{
    if(!out_state.invHessGrad.empty()) {
        for(uint32_t i = 0; i != m_ttl; ++i)
            m_ep[i] = m_p[i] * (1.0 - m_p[i]);

        //EigenMapMatCol mData(m_vecData.data(), m_ttl, m_Dimension);
        EigenMapVec mepVec(m_ep.data(), m_ttl);
        EigenMat mHess(m_Dimension, m_Dimension);
        mHess.noalias() = mData.transpose() * mepVec.asDiagonal() * mData;

        dense_sparse_matrix::matrixProductATWA<EigenMapVec>(m_vecData, mepVec,  mHess);

        for(size_t i=0; i<m_Dimension; ++i)
        {
            for(size_t j=0; j<m_Dimension; ++j)
            {
                mHess(i, j) /= m_ttl;
            }
        }
        for(size_t i=1; i<m_Dimension; ++i)
        {
            mHess(i, i) += m_c;
        }

        EigenMapVec b(out_state.grad.data(), m_Dimension);
        EigenMapVec answer(out_state.invHessGrad.data(), m_Dimension);
        answer = mHess.ldlt().solve(b);

    }

    return 0;
}

// calculate log(x) and exp(x) in parallel: log and exp calculation cost much time
void ObjectFunction::calculateLogExpByParallel(const EigenMapArr& thetaxArr, const EigenMapArr& catArr,
                    size_t threadNumber, ltt::allocator* pAlloc,
                    double& sum, EigenMapArr& pArr)
{
    sum = 0.0;
    size_t size_ = thetaxArr.rows();
    const size_t MIN_SIZE_TO_PARALLEL = 100000;
    if(size_ < MIN_SIZE_TO_PARALLEL || threadNumber == 1){ // single thread

       Eigen::ArrayXd logexpArr(size_);

       pArr = exp(-thetaxArr.abs());
       logexpArr = (thetaxArr+thetaxArr.abs())/2 + log1p(pArr) - catArr * thetaxArr;
       pArr = (1 + pArr).inverse();
       for(uint32_t i=0; i != size_; ++i){
           if(thetaxArr(i) < 0){
               pArr(i) = 1.0 - pArr(i);
           }
       }
       sum = logexpArr.matrix().sum();

    }
    else{ // multi-thread
        size_t finalThreadNumber = static_cast<size_t>(getSysNumsCore());
        if(threadNumber < finalThreadNumber) finalThreadNumber = threadNumber;
        if(finalThreadNumber > size_) finalThreadNumber = size_;

        TRexUtils::Parallel::Context context;
        size_t step = size_ / finalThreadNumber;
        ltt::vector<size_t> pStart(finalThreadNumber, 0, *pAlloc);
        size_t remainder = size_ - step * finalThreadNumber;
        for(size_t i = 0; i < remainder; ++i)
            pStart[i] = (step + 1) * i;
        for(size_t i = remainder; i < finalThreadNumber; ++i)
            pStart[i] = remainder + step * i;

        sum = 0.0;

        _STL::vector<double> resVec(finalThreadNumber, 0);

        TRexUtils::Parallel::SimpleBlockPartitioner<size_t>  m_pPartitioner(0, size_, 128);

        for(size_t j = 0; j < finalThreadNumber - 1; ++j)
            context.pushJob(new(*pAlloc) calculateSumJob(m_pPartitioner, thetaxArr, catArr, resVec[j], pArr));

        context.run();

        //aggregate result
        for(size_t i=0; i<finalThreadNumber; ++i){
            sum += resVec[i];
        }

    }

}
}// end of LOGR
}// end of AFL_PAL
