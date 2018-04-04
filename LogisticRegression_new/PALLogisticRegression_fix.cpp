/*!
 * PALLogisticRegression_fix.cpp
 *
 *  Created on: March 12, 2018
 *      Author: I323463
 */
#include "AFL/PAL/Classification/LogisticRegression_new/PALLREnetRegularizer_new.h"
#include "AFL/PAL/Classification/LogisticRegression_new/PALLogisticRegression_fix.h"
#include "AFL/PAL/DataPreProcessing/Transform2Integer/Attri2IntBase.h"
#include "AFL/PAL/DataPreProcessing/Transform2Integer/Attri2IntRep.h"
#include "AFL/PAL/DataPreProcessing/Transform2Integer/Attri2IntVisitor.h"
#include "AFL/PAL/DataPreProcessing/Transform2Integer/ContAttri2Int.h"
#include "AFL/PAL/DataPreProcessing/Transform2Integer/DisAttri2Int.h"
#include "AFL/PAL/ModelEvaluation/Utilities/PALModelEvaluateUtility.h"
#include "AFL/PAL/PALTableTypeVerifier.h"
#include "AFL/PAL/Statistics/CommonFunctions/PValueCalc.hpp"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include "AFL/PAL/Utilities/Optimization/Solver.hpp"
#include "AFL/PAL/Utilities/PMML/palGenPmml.h"
#include "AFL/PAL/Utilities/PMML/regressionModel.h"
#include "AFL/PAL/palCancel.h"
#include "AFL/PAL/palCommonFunction.hpp"
#include "AFL/PAL/palParameters.h"
#include "AFL/PAL/palTableTypes.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataLoader.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataStruct.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/ModelTrainer.h"
#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/ModelEvaluator.h"

#include <Basis/Random.hpp>
#include <TRexUtils/BitVector.h>
#include <TRexUtils/ParallelDispatcher.h>
#include <TRexUtils/Profiler.h>
#include <ltt/adp/cmath.hpp>
#include <ltt/auto_ptr.hpp>
#include <ltt/error_registry.hpp>
#include <ltt/ext/rng.hpp>
#include <ltt/strstream.hpp>
#include <ltt/vector.hpp>
#include <chrono>

USE_ERROR(AFL_PAL, ERR_PAL_CLASSIFY_INVALID_VALUE);
USE_ERROR(AFL_PAL, ERR_PAL_CLASSIFY_SHORT_DATA);
USE_ERROR(AFL_PAL, ERR_PAL_COMM_INTERNAL_ERR);
USE_ERROR(AFL_PAL, ERR_PAL_COMM_NULL_VALUES_IN_TABLE);
USE_ERROR(AFL_PAL, ERR_PAL_COMM_TABLE_COL_SIZE);
USE_ERROR(AFL_PAL, ERR_PAL_COMM_TABLE_EMPTY);
USE_ERROR(AFL_PAL, ERR_PAL_INVALID_COLUMN_TYPE);
USE_TRACE(TRACE_PAL_LOGISTICREGRESSION);
USE_ERROR(AFL_PAL, ERR_PAL_PREPROCESS_PARAM_INVALID);
USE_TRACE(TRACE_PAL_TEST);
USE_ERROR(AFL_PAL, ERR_PAL_COMM_PARAMETER_INVALID);

using namespace TRexCommonObjects;
using TRexUtils::Parallel::JobResult;
using TRexUtils::Parallel::JobBase;
#define EPSINON  1E-8

namespace AFL_PAL
{
namespace LOGR
{

using ltt::auto_ptr;
using TrexTypes::DoubleAttributeValue;

// the entrance of Logistic Regression
void PALLogisticRegression_new::execute()
{
    PROF_SCOPE("Profiling LogisticRegression function");
    TRACE_INFO(TRACE_PAL_LOGISTICREGRESSION,
            "Information: PALLogisticRegression::execute() begin" << ltt::endl);

    // check the type validity of the internal table.
    TableVerifier tv("[IdS]+");
    tv.verify(dataTab);
    tv.setRegExp(TYPE_PAL_PARAMETER);
    tv.verify(iargsTab);
    tv.setRegExp("[IS][Id]+");
    tv.verify(coefficientTab);
    tv.setRegExp("IS");
    tv.verify(pmmlTab);

    // get the needed parameters
    getArguments();

    // check the validity of the input data
    checkAndValidateTables();

    /// category_col && categorical_variable
    uint32_t attNum = dataTab->getNumColumns() - 1;
    ltt_adp::vector<bool> isCategoryVec(attNum, false);
    ltt_adp::vector<bool> categoryColVec(attNum, false);

    getCategoryVec(isCategoryVec, categoryColVec);

    if (outputCoeByKey == true){
        getCoefficientKey(coefficientKey, isCategoryVec);
    }

    //Eigen::VectorXd label(dataTab->size());
    //loadLabel(label);

    //dense_sparse_matrix::DSMatrix densesparseMat(*hAlloc);
    //ltt_adp::vector<uint32_t> categoryNumofEachColumn;
    //ltt_adp::vector<uint32_t> insPos;
    // load data into DSMatrix from InternalTable
    //loadDataFromInternalTable(dataTab, categoryColVec, categoryNumofEachColumn, insPos, densesparseMat);

    // standardize data
    //if(LR_Args.standardize == 1)
    //    densesparseMat.standardize(m_meanVec, m_stdVec);

    //int finalIter = 0;
    //Eigen::VectorXd coefficient(densesparseMat.cols());
    //ENetRegularizer_new::lrPGDSolver(densesparseMat, label, LR_Args.lambda, LR_Args.alpha,
    //                             LR_Args.maxIteration, LR_Args.threadNumber,
    //                             finalIter, coefficient, LR_Args.threshold, hAlloc.get());

    //size_t contNum = densesparseMat.denseMat.cols();
    //ltt::vector<double> backCoeff(coefficient.rows(),*hAlloc);

    //uint32_t expandColSize, feature_dimension;
    //double objValue, minusMLE;
    //calculateObjAndMLE(densesparseMat, label, coefficient, expandColSize, feature_dimension, objValue, minusMLE);

    //int solutionStatus = 0;
    //outputStatistic(minusMLE, feature_dimension, objValue, finalIter, solutionStatus);

    //if(LR_Args.standardize == 1)
    //    retBackCoefficient(coefficient, contNum);

    //ltt::vector<double> zscore(*hAlloc);
    //ltt::vector<double> pvalue(*hAlloc);

    //if(LR_Args.pvalue == 1){
    //    if(LR_Args.standardize == 1)
    //        densesparseMat.restoreMatrix(m_meanVec, m_stdVec);
    //    calculatePValue(densesparseMat, coefficient, insPos, zscore, pvalue);
    //}

    //sortCoefficient(coefficient, categoryNumofEachColumn, insPos, contNum, backCoeff, expandColSize);

    //writeCoefficientToTable(backCoeff, insPos, expandColSize, pvalue, zscore);

}

//void PALLogisticRegression_new::retBackCoefficient(Eigen::VectorXd& orgCoeff, size_t& contNum)
//{
//
//    double tmpVal = 0.0;
//    for(size_t contIndex=0; contIndex<contNum; ++contIndex){
//        if(LR_Args.standardize != 0 && m_stdVec[contIndex] > 0){
//            tmpVal += orgCoeff(contIndex)*m_meanVec[contIndex]/m_stdVec[contIndex];
//            orgCoeff(contIndex) /= m_stdVec[contIndex];
//        }
//    }
//    orgCoeff(0) -= tmpVal;
//}
//
//void PALLogisticRegression_new::calculatePValue(dense_sparse_matrix::DSMatrix& dsMat, Eigen::VectorXd& coeff_NewValue,
//                                              ltt_adp::vector<uint32_t>& insPos, ltt::vector<double>& zscore,
//                                              ltt::vector<double>& pvalue)
//{
//   uint32_t ttl = dsMat.rows();
//   uint32_t mDimension = coeff_NewValue.rows();
//   Eigen::VectorXd p(ttl);
//   ltt::vector<double> StError(*hAlloc);
//   dense_sparse_matrix::DSMatrix::matrixVectorProduct(dsMat, coeff_NewValue, p, hAlloc.get(), LR_Args.threadNumber);
//
//   for(uint32_t i=0; i < ttl; ++i){
//       double tmp = 1.0/(1+exp(-p(i)));
//       p(i) = tmp * (1-tmp);
//   }
//
//   Eigen::MatrixXd hess(mDimension, mDimension);
//   dense_sparse_matrix::DSMatrix::matrixProductATWA(dsMat, p, hess);
//   MatrixdCol matrixHess(hess.data(), mDimension, mDimension);
//
//   double delta = 1e-6;
//   StError.clear();
//
//    // calculate StandardError
//    SVD<MatrixdCol, SVDEigen> svd(matrixHess);
//    
//    try
//    {
//        const MatrixdCol& invHess = svd.matrixInverse();
//
//        for (uint32_t i = 0; i < mDimension; ++i)
//        {
//            if (invHess(i, i) > 0)
//            {
//                StError.push_back(sqrt(invHess(i, i)));
//            }
//            else
//            {
//                StError.push_back(0.0);
//            }
//        }
//    }
//    catch(const ltt::exception&)
//    {
//        for (uint32_t i = 0; i < mDimension; ++i)
//        {
//            matrixHess(i, i) += delta;
//        }
//
//        const MatrixdCol& invHess = svd.matrixInverse();
//
//        for (uint32_t i = 0; i < mDimension; ++i)
//        {
//            if (invHess(i, i) > 0)
//            {
//                StError.push_back(sqrt(invHess(i, i)));
//            }
//            else
//            {
//                StError.push_back(0.0);
//            }
//        }
//    }
//
//    zscore.clear();
//    zscore.resize(mDimension);
//
//    for (uint32_t i = 0; i != mDimension; ++i)
//    {
//        if (fabs(StError.at(i)) < 1e-6)
//        {
//            zscore.at(i) = 0;
//            pvalue.push_back(-1);  // P-value is invalid.
//        }
//        else
//        {
//            zscore.at(i) = coeff_NewValue(i) / StError.at(i);
//            pvalue.push_back(2 * STAT::pvalueNormalDist(fabs(zscore.at(i))));
//        }
//
//    }
//
//    // insert value
//    for (uint32_t i = 0; i != insPos.size(); ++i)
//    {
//        zscore.insert(zscore.begin() + insPos.at(i), 0.0);
//        pvalue.insert(pvalue.begin() + insPos.at(i), -1);
//    }
//
//}
//
//void PALLogisticRegression_new::calculateObjAndMLE(dense_sparse_matrix::DSMatrix& dsMat, Eigen::VectorXd& yValue,
//                                   Eigen::VectorXd& coeff, uint32_t& expandColSize,
//                                   uint32_t& feature_dimension, double& objValue, double& minusMLE)
//{
//    uint32_t ncol = dsMat.cols();
//    uint32_t nrow = dsMat.rows();
//    feature_dimension = ncol;
//    expandColSize = ncol - 1;
//    objValue = 0;
//    minusMLE = 0;
//
//    Eigen::VectorXd Xb(nrow);
//    dense_sparse_matrix::DSMatrix::matrixVectorProduct(dsMat, coeff, Xb, hAlloc.get(), LR_Args.threadNumber);
//
//    for (uint32_t i = 0; i < nrow; ++i){
//        if(yValue[i] == -1)
//            yValue[i] = 0;
//        if(Xb[i] < 0)
//            minusMLE += log1p(exp(Xb[i])) - Xb[i] * yValue[i];
//        else if(Xb[i] >= 0)
//            minusMLE += Xb[i] + log1p(exp(-Xb[i])) - Xb[i] * yValue[i];
//    }
//
//    objValue = minusMLE / nrow;
//    double sumOfOneNorm = 0.0;
//    double sumOfTwoNorm = 0.0;
//    for(uint32_t i = 0; i < expandColSize; i++){
//        sumOfOneNorm += fabs(coeff[i+1]);
//        sumOfTwoNorm += (coeff[i + 1]) * (coeff[i + 1]);
//    }
//
//    objValue += LR_Args.lambda * (LR_Args.alpha * sumOfOneNorm + 1.0 * (1 - LR_Args.alpha)/2 * sumOfTwoNorm);
//
//    for (uint32_t i =0; i < expandColSize + 1; ++i ){
//        if ( fabs(coeff[i]) < COMMON_DOUBLE_ZERO )
//            --feature_dimension;
//    }
//
//}
//
//void PALLogisticRegression_new::sortCoefficient(Eigen::VectorXd& orgCoeff, ltt_adp::vector<uint32_t>& cateNumofEachColumn,
//                ltt_adp::vector<uint32_t>& insPos, size_t contNum, ltt::vector<double>& backCoeff, uint32_t& expandColSize)
//{
//    backCoeff[0] = orgCoeff(0);
//    size_t cateIndex = 0;
//    size_t contIndex = 0;
//    size_t totalIndex = 0;
//
//    double tmpVal = 0.0;
//    for(size_t i=0; i<cateNumofEachColumn.size(); ++i){
//        if(cateNumofEachColumn[i] <= 0){ // continuous column
//            totalIndex += 1;
//            contIndex += 1;
//            //if(LR_Args.standardize != 0 && m_stdVec[contIndex] > 0){
//            //    tmpVal += orgCoeff(contIndex)*m_meanVec[contIndex]/m_stdVec[contIndex];
//            //    orgCoeff(contIndex) /= m_stdVec[contIndex];
//            //}
//            backCoeff[totalIndex] = orgCoeff(contIndex);
//        }
//        else{ // categorical column
//            for(uint32_t j=0; j<cateNumofEachColumn[i]-1; ++j){
//                totalIndex +=1;
//                backCoeff[totalIndex] = orgCoeff(contNum+cateIndex);
//                cateIndex += 1;
//            }
//        }
//    }
//    backCoeff[0] -= tmpVal;
//
//    for(size_t i=0; i<insPos.size(); ++i){
//        backCoeff.insert(backCoeff.begin() + insPos.at(i), 0.0);
//    }
//
//}
//
//void PALLogisticRegression_new::writeCoefficientToTable(ltt::vector<double>& coefficient, ltt::vector<uint32_t>& insPos,
//                              uint32_t& expandColSize, ltt::vector<double>& pvalue, ltt::vector<double>& zscore)
//{
//    expandColSize += insPos.size();
//
//    // put the value into the coefficient Table
//    TRexCommonObjects::ColumnBase *coefficientIDCol =
//        coefficientTab->getColumn(0);
//    TRexCommonObjects::ColumnBase *coefficientValueCol =
//        coefficientTab->getColumn(1);
//
//    for (uint32_t beta = 0; beta < expandColSize + 1; ++beta)
//    {
//        if (outputCoeByKey == true)
//        {
//            coefficientIDCol->set(beta, coefficientKey[beta]);
//        }
//        else
//        {
//            coefficientIDCol->set(beta, beta);
//        }
//
//        coefficientValueCol->set(beta, coefficient[beta]);
//    }
//
//    int colSize = coefficientTab->getNumColumns();
//
//    if (1 == LR_Args.pvalue)
//    {
//        TRexCommonObjects::ColumnBase *coefficientZscoreCol =
//            coefficientTab->getColumn(2);
//        TRexCommonObjects::ColumnBase *coefficientPValueCol =
//            coefficientTab->getColumn(3);
//
//        for (uint32_t beta = 0; beta < expandColSize + 1; ++beta)
//        {
//            coefficientZscoreCol->set(beta, zscore[beta]);
//            coefficientPValueCol->set(beta, pvalue[beta]);
//        }
//
//        // remove the invalid p-value
//        for (uint32_t beta = 0; beta < expandColSize + 1; ++beta)
//        {
//            if (fabs(pvalue[beta] + 1) < EPSINON)
//            {
//                coefficientZscoreCol->erase(beta);
//                coefficientPValueCol->erase(beta);
//            }
//        }
//    }
//    else if( (0 == LR_Args.pvalue) && (colSize == 4) )
//    {
//        TRexCommonObjects::ColumnBase *coefficientZscoreCol =
//            coefficientTab->getColumn(2);
//        TRexCommonObjects::ColumnBase *coefficientPValueCol =
//            coefficientTab->getColumn(3);
//
//        coefficientZscoreCol->resize(expandColSize + 1);
//        coefficientPValueCol->resize(expandColSize + 1);
//    }
//
//}
//
//void PALLogisticRegression_new::getCategoryVec(ltt_adp::vector<bool> &isCategoryVec,
//                                           ltt_adp::vector<bool>& categoryColVec)
//{
//    int len = isCategoryVec.size();
//    PalParameters<ltt_adp::string> pps(iargsTab);
//    _STL::vector<uint32_t> vecRowIndex(*hAlloc);
//    _STL::string name("CATEGORY_COL", *hAlloc);
//    ltt_adp::vector<ltt_adp::string> category_col_name;
//    if(pps.registerParameterVec("CATEGORICAL_VARIABLE"))
//    {
//
//        ltt_adp::vector<ltt_adp::string> const * pDiscColNames = &(pps.getParameterVec("CATEGORICAL_VARIABLE"));
//        if(pDiscColNames)
//           category_col_name.assign(pDiscColNames->begin(), pDiscColNames->end());
//
//        for(int colindex=0; colindex<len; colindex++)
//        {
//            if ( (dataTab->getColumn(colindex)->getValueType() == TRexEnums::AttributeType::INT) &&
//                 (ltt_adp::find(category_col_name.begin(),category_col_name.end(),
//                    dataTab->getColumn(colindex)->getName()) != category_col_name.end()))
//            {
//                isCategoryVec[colindex] = true;
//            }
//        }
//    }
//    if(category_col_name.empty()) // pps.checkItem("CATEGORY_COL")
//    {
//        const ColumnBase *nameCol = iargsTab->getColumn(0);
//        const ColumnBase *intCol = iargsTab->getColumn(1);
//        if ( nameCol->find(name, vecRowIndex) )
//        {
//            _STL::vector<uint32_t>::const_iterator itr =vecRowIndex.begin(), end_itr = vecRowIndex.end();
//            int colIndex = 0;
//            for (; itr!=end_itr; ++itr)
//            {
//                if ( !intCol->get(*itr, colIndex) )
//                {
//                    THROW_PAL_ERROR(ERR_PAL_PREPROCESS_PARAM_INVALID, ltt::msgarg_text("COL", intCol->getName()) << ltt::msgarg_int("ROW", static_cast<int>(*itr+1)) );
//                }
//                if(LR_Args.hasId == 1) colIndex -= 1;
//                if (colIndex<0 || colIndex>=len) continue;
//
//                if (dataTab->getColumn(colIndex)->getValueType() == TRexEnums::AttributeType::INT)
//                {
//                    isCategoryVec[colIndex] = true;
//                }
//
//           } //for (itr!=end_itr)
//        }
//     }// if (nameCol->find(...))
//
//     for(int i=0; i<len; ++i) // add string column into categoryColVec
//     {
//         categoryColVec[i] = isCategoryVec[i];
//         if(dataTab->getColumn(i)->getValueType() == TRexEnums::AttributeType::STRING)
//         {
//             categoryColVec[i] = true;
//         }
//     }
//
//}
//
//void PALLogisticRegression_new::getCoefficientKey(ltt::vector<_STL::string>& key,
//                                              ltt_adp::vector<bool> isCategoryVec)
//{
//    TRexCommonObjects::ColumnBase *col = NULL;
//    TRexEnums::AttributeType idType;
//    bool is_string_type = true;
//
//    key.clear();
//    key.push_back(INTERCEPT);
//
//    // collect the category column's set via input parameter
//    if (categoryColSet.empty() == true)
//    {
//
//        int len = isCategoryVec.size();
//        for(int i=0; i<len; i++)
//            if(isCategoryVec[i])
//                categoryColSet.insert(i);
//    }
//
//    for (uint32_t i = 0; i < dataTab->getNumColumns() - 1; ++i)
//    {
//        is_string_type = false;
//        col = dataTab->getColumn(i);
//
//        idType = col->getTypeDefinition().getAttributeType();
//
//        if (idType == TRexEnums::AttributeType::STRING ||
//                idType == TRexEnums::AttributeType::CLOB ||
//                idType == TRexEnums::AttributeType::NCLOB)
//        {
//            is_string_type = true;
//        }
//
//        // If categorical type then find all the values
//        if (is_string_type == true ||
//                categoryColSet.find(i) != categoryColSet.end())
//        {
//            // col loop
//            if (is_string_type == true)
//            {
//                ltt_adp::set<_STL::string> valueSet;
//                ltt_adp::set<_STL::string>::iterator iter;
//                _STL::string value;
//
//                for (uint32_t j = 0; j < col->size(); ++j)
//                {
//                    // row loop
//                    col->get(j, value);
//
//                    // cope with empty strings
//                    if (value.empty())
//                    {
//                         value = EMPTY_STRING;
//                    }
//                    iter = valueSet.find(value);
//
//                    if (iter == valueSet.end())  // new value found
//                    {
//                        key.push_back(col->getName() + DELIMIT + value);
//                        valueSet.insert(value);
//                    }
//                }
//            }
//            else if (idType == TRexEnums::AttributeType::INT)
//            {
//                ltt_adp::set<int32_t> valueSet;
//                ltt_adp::set<int32_t>::iterator iter;
//                int32_t value;
//
//                for (uint32_t j = 0; j < col->size(); ++j)
//                {
//                    // row loop
//                    col->get(j, value);
//                    iter = valueSet.find(value);
//
//                    if (iter == valueSet.end())  // new value found
//                    {
//                        _STL::stringstream ss(ltt::ios_base::goodbit);
//                        ss << value;
//                        key.push_back(col->getName() + DELIMIT + ss.str());
//                        valueSet.insert(value);
//                    }
//                }
//            }
//        }
//        else
//        {
//            key.push_back(col->getName());
//        }
//    }
//}
//
//void PALLogisticRegression_new::getArguments()
//{
//    PalParameters<int32_t> ppi(iargsTab);
//    ppi.registerParameter("HAS_ID", 0,
//            new IntRangeCheck(0, 1, true, true)); // check if has id
//    LR_Args.hasId = ppi["HAS_ID"]; // check if has id
//
//    if(LR_Args.hasId == 1)
//    {
//        dataTab->removeColumn(uint32_t(0));  // if has id, then remove first column
//    }
//
//    ppi.registerParameter("VARIABLE_NUM", dataTab->getNumColumns() - 1,
//            new IntRangeCheck(0, dataTab->getNumColumns() - 1, false, true));
//    // THREAD_RATIO
//    PalParameters<double> ppd(iargsTab);
//    size_t numthread = static_cast<size_t> (getSysNumsCore());
//    if(ppd.checkItem("THREAD_RATIO"))
//    {
//        LR_Args.threadNumber = numthread; // backward compatibility default value
//        ppd.registerParameter("THREAD_RATIO", 1.0);
//        double threadRatio = ppd["THREAD_RATIO"];
//        if(fabs(threadRatio - 0.0) < EPSINON)
//            LR_Args.threadNumber = 1;
//        else if(threadRatio > 0.0 && threadRatio <= 1.0)
//            LR_Args.threadNumber = ceil(threadRatio * getSysNumsCore());
//    }
//    else
//    {
//        ppi.registerParameter("THREAD_NUMBER", numthread,
//                new IntRangeCheck(1, MAXTHREAD, true, true));
//        LR_Args.threadNumber = ppi["THREAD_NUMBER"];
//    }
//    ppi.registerParameter("PMML_EXPORT", 0,
//            new IntRangeCheck(0, 2, true, true));
//    ppi.registerParameter("METHOD", 0,
//            new IntRangeCheck(0, 8, true, true));
//    ppi.registerParameter("STAT_INF", 0,
//            new IntRangeCheck(0, 1, true, true));
//    ppi.registerParameter("SEARCH_STRATEGY", 0, new IntRangeCheck(0, 1, true, true));
//    LR_Args.useGridSearch = (ppi["SEARCH_STRATEGY"] == 0);
//
//    // exclusive parameters for SGD
//    ppi.registerParameter("SGD_TYPE", 0,
//            new IntRangeCheck(0, 1, true, true));  // not exposed
//    //ppi.registerParameter("LEARNING_RATE_TYPE", 0,
//    //      new IntRangeCheck(0, 1, true, true));
//    ppi.registerParameter("SEED", 0,
//            new IntBtCheck(0, true));
//    ppi.registerParameter("FORCE_PASS", 1,
//            new IntRangeCheck(0, 1, true, true));  // not exposed
//    ppi.registerParameter("MAX_PASS_NUMBER", 1,
//            new IntRangeCheck(1, 100, true, true));
//    ppi.registerParameter("STANDARDIZE", 1, new IntRangeCheck(0, 1, true, true));
//
//    // exclusive parameters for LBFGS
//    ppi.registerParameter("LBFGS_M", 6,
//            new IntBtCheck(3, true));
//
//    LR_Args.betaNumber   = ppi["VARIABLE_NUM"];
//    LR_Args.pmmlFlag     = ppi["PMML_EXPORT"];
//    LR_Args.method       = ppi["METHOD"];
//    LR_Args.pvalue = ppi["STAT_INF"];
//
//    LR_Args.sgdType = ppi["SGD_TYPE"];
//    LR_Args.forcePass = ppi["FORCE_PASS"];
//    LR_Args.passNum = ppi["MAX_PASS_NUMBER"];
//    //LR_Args.learningRateType = ppi["LEARNING_RATE_TYPE"];
//    LR_Args.seed = ppi["SEED"];
//    LR_Args.lbfgsM = ppi["LBFGS_M"];
//    LR_Args.standardize = ppi["STANDARDIZE"];
//
//    ppd.registerParameter("ENET_ALPHA", 1.0,
//            new DoubleRangeCheck(0.0, 1.0, true, true));
//    LR_Args.alpha = ppd["ENET_ALPHA"];
//
//    if(m_enableParamSelection != true)
//    {
//        ppd.registerParameter("ENET_LAMBDA", 0.0,
//                new DoubleBtCheck(0.0, true));
//        LR_Args.lambda = ppd["ENET_LAMBDA"];
//    }
//    //if(LR_Args.lambda>0.0) {/*nothing*/}
//    //else if((LR_Args.method == 2 || LR_Args.method == 6) && !m_enableParamSelection)
//    //{
//    //    LR_Args.method = 3; // if lambda is equal to 0 and method is not 0 or 3, then use lbfgs method by default
//    //}
//    //if((LR_Args.lambda>0) && LR_Args.alpha>0 && (LR_Args.method!=2 && LR_Args.method!=6))
//    //{
//    //    LR_Args.method = 6; // if penalty term include l1 term and alg is not CD or PGD, then using PGD by default.
//    //}
//    //LR_Args.c = LR_Args.lambda;
//
//    // Exclusive parameters for Elastic-net regularization
//    // PalParameters<double> ppd(iargsTab);
//    ppd.registerParameter("ALPHA", 1.0,
//            new DoubleBtCheck(0.0, false));  // not exposed
//    ppd.registerParameter("GAMMA", 1.0,
//            new DoubleBtCheck(0.0, false));  // not exposed
//    ppd.registerParameter("C", 0.5,
//            new DoubleBtCheck(0.0, false));  // not exposed
//
//    LR_Args.sgdAlpha = ppd["ALPHA"];
//    LR_Args.sgdGamma = ppd["GAMMA"];
//    LR_Args.sgdC = ppd["C"];
//
//    if (2 == LR_Args.method || 6 == LR_Args.method)
//    {
//
//        ppd.registerParameter("EXIT_THRESHOLD", 1.0e-7,
//                new DoubleBtCheck(0.0, false));
//    if(2 == LR_Args.method)
//            ppi.registerParameter("MAX_ITERATION", 1e5,
//                    new IntBtCheck(0, false));
//    else if(6 == LR_Args.method)
//            ppi.registerParameter("MAX_ITERATION", 1000,
//                    new IntBtCheck(0, false));
//    }
//    else
//    {
//        ppd.registerParameter("EXIT_THRESHOLD", 1.0e-6,
//                new DoubleBtCheck(0.0, false));
//        ppi.registerParameter("MAX_ITERATION", 100,
//                new IntBtCheck(0, false));
//        if(LR_Args.method == 0) // ConvLR: xtol: default is 1e-6
//        {
//            ppd.registerParameter("EPSILON", 1e-6,
//                    new DoubleBtCheck(0.0, false));
//            LR_Args.epsilon = ppd["EPSILON"];
//        }
//        else if(LR_Args.method == 3 || LR_Args.method == 5) // ConvStrict: epsilon default is 1e-5
//        {
//            ppd.registerParameter("EPSILON", 1e-5,
//                    new DoubleBtCheck(0.0, false));
//            LR_Args.epsilon = ppd["EPSILON"];
//        }
//    }
//
//    LR_Args.threshold = ppd["EXIT_THRESHOLD"];
//    LR_Args.maxIteration = ppi["MAX_ITERATION"];
//
//    PalParameters<_STL::string> pps(iargsTab);
//    pps.registerParameter("SELECTED_FEATURES", "");
//    pps.registerParameter("DEPENDENT_VARIABLE", "");
//    LR_Args.featureStr = pps["SELECTED_FEATURES"];
//    LR_Args.depVar     = pps["DEPENDENT_VARIABLE"];
//
//}
//
//void PALLogisticRegression_new::composeDataTable(
//        TRexCommonObjects::InternalTable* featureTable,
//        TRexCommonObjects::ColumnBase* target)
//{
//    if (featureTable == NULL)
//    {
//        ltt::smartptr_handle<TRexCommonObjects::InternalTable> tmp(static_cast<TRexCommonObjects::InternalTable *>(dataTab->clone()));
//        mDataTabPtr = tmp;
//        dataTab = mDataTabPtr.get();
//
//        PalParameters<int> ppi(iargsTab);
//        ppi.registerParameter("HAS_ID", 0, new IntRangeCheck(0, 1, true, true));
//        int hasId = ppi["HAS_ID"];
//        _STL::string firstColumnName = dataTab->getColumn(0)->getName();
//
//        _STL::string name1 = target->getName();
//        if(hasId == 1)
//        {
//            if(firstColumnName == name1)
//                THROW_PAL_ERROR(ERR_PAL_COMM_INTERNAL_ERR,
//                        ltt::msgarg_text("APPEND", " ID column can not be set to the DEPENDENT_VARIABLE"));
//        }
//
//        int32_t index = 0;
//        _STL::string colName = target->getName();
//        //ltt::cout << "colName  " << colName << ltt::endl;
//        //find the corresponding column index
//        for (uint32_t i = 0; i < dataTab->getNumColumns(); ++i)
//        {
//            ltt_adp::string columnName = dataTab->getColumn(i)->getName();
//            if (columnName == colName)
//            {
//                index = i;
//                break;
//            }
//        }
//
//        //ltt::cout << "colName Index  " << index << ltt::endl;
//
//        ColumnBase* nameCol = iargsTab->getColumn(0);
//        ColumnBase* intCol = iargsTab->getColumn(1);
//        uint32_t pos = 0;
//        uint32_t size = nameCol->size();
//
//        // Check whether the category column needs to be adjusted
//        while((pos = nameCol->findNext(pos, _STL::string("CATEGORY_COL"))) < size)
//        {
//            int32_t value;
//            intCol->get(pos, value);
//            if(value == index)
//            {
//                THROW_PAL_ERROR(ERR_PAL_COMM_INTERNAL_ERR, ltt::msgarg_text("EXCEPT", "DEPENDENT_VARIABLE cannot be set to CATEGORY VARIABLE."));
//            }
//            if(value > index)
//                intCol->set(pos, value - 1);
//
//            ++pos;
//
//        }
//
//        dataTab->removeColumn(target->getName());  // Remove the target first and later add it to the end
//    }
//    else
//    {
//        // Clone the control table for modification
//        ltt::smartptr_handle<TRexCommonObjects::InternalTable> tmp(static_cast<TRexCommonObjects::InternalTable *>(iargsTab->clone()));
//        mArgsTabPtr = tmp;
//        iargsTab = mArgsTabPtr.get();
//
//        _STL::string colName;
//        uint32_t i;
//        ColumnBase* nameCol = iargsTab->getColumn(0);
//        ColumnBase* intCol = iargsTab->getColumn(1);
//        uint32_t pos = 0;
//        uint32_t size = nameCol->size();
//        bool find = false;
//
//        // Check whether the category column needs to be adjusted
//        while((pos = nameCol->findNext(pos, _STL::string("CATEGORY_COL"))) < size)
//        {
//            int32_t value;
//            intCol->get(pos, value);
//
//            find = false;
//            colName = dataTab->getColumn(value)->getName();
//
//            for (i = 0; i < featureTable->getNumColumns(); ++i)
//            {
//                if (colName == featureTable->getColumn(i)->getName())
//                {
//                    // Adjust according to the column index in the feature table
//                    intCol->set(pos, i);
//                    find = true;
//                    categoryColSet.insert(i);
//                    break;
//                }
//            }
//
//            // The column is not selected for processing
//            if (find == false)
//            {
//                iargsTab->removeRow(pos);
//                --size;
//            }
//            else
//            {
//                ++pos;
//            }
//        }
//
//        dataTab = featureTable;
//        PalParameters<int> ppi(iargsTab);
//        ppi.registerParameter("HAS_ID", 0, new IntRangeCheck(0, 1, true, true));
//        int hasId = ppi["HAS_ID"];
//        if(hasId == 1)
//        {
//            ColumnBase * idCol = dataTab->getColumn(0);
//            dataTab->insertColumn(0, idCol);
//        }
//    }
//
//    dataTab->addColumn(target);
//}
//
//void PALLogisticRegression_new::outputStatistic(double& minusMLE, uint32_t& feature_dimension, double& objValue,
//                        int32_t& finalIter, int& solutionStatus)
//{
//    double AIC = (-2) * (-minusMLE) + 2*feature_dimension;
//
//    TRexCommonObjects::ColumnBase* statNameCol = statisticsTab->getColumn(0);
//    TRexCommonObjects::ColumnBase* statValueCol = statisticsTab->getColumn(1);
//
//    statNameCol->set(0, "AIC");
//    statValueCol->set(0, AIC);
//
//    minusMLE = -1 * minusMLE;
//
//    statNameCol->set(1, "obj");
//    statValueCol->set(1, objValue);
//    statNameCol->set(2, "log-likelihood");
//    statValueCol->set(2, minusMLE);
//    statNameCol->set(3, "iter");
//    statValueCol->set(3, finalIter);
//
//    switch(LR_Args.method)
//    {
//        case 0:
//        {
//            statNameCol->set(4, "method");
//            statValueCol->set(4, "newton");
//            break;
//        }
//        case 1:
//        {
//            statNameCol->set(4, "method");
//            statValueCol->set(4, "gradient descent");
//            break;
//        }
//        case 2:
//        {
//            statNameCol->set(4, "method");
//            statValueCol->set(4, "cyclical coordinate descent");
//            break;
//        }
//        case 3:
//        {
//            statNameCol->set(4, "method");
//            statValueCol->set(4, "lbfgs");
//            break;
//        }
//        case 4:
//        {
//            statNameCol->set(4, "method");
//            statValueCol->set(4, "stochastic gradient descent");
//            break;
//        }
//        case 5:
//        {
//            statNameCol->set(4, "method");
//            statValueCol->set(4, "bfgs");
//            break;
//        }
//        case 6:
//        {
//            statNameCol->set(4, "method");
//            statValueCol->set(4, "proximal gradient descent");
//            break;
//        }
//    }
//
//    if(LR_Args.method != 2 && LR_Args.method != 4 && LR_Args.method != 6)
//    {
//        if(solutionStatus == 0)
//        {
//            statNameCol->set(5, "solution status");
//            statValueCol->set(5, "Converged");
//        }
//        else if(solutionStatus == 1 || solutionStatus == 2)
//        {
//            statNameCol->set(5, "solution status");
//            statValueCol->set(5, "Convergence not reached, line search failed");
//        }
//        else if(solutionStatus == 3)
//        {
//            statNameCol->set(5, "solution status");
//            statValueCol->set(5, "Convergence not reached after maximum number of iterations");
//        }
//        else if(solutionStatus == 5)
//        {
//            statNameCol->set(5, "solution status");
//            statValueCol->set(5, "Convergence not reached, update direction failed");
//        }
//    }
//    if(LR_Args.method == 6 || LR_Args.method == 2)
//    {
//        if((int32_t)finalIter == (LR_Args.maxIteration))
//        {
//            statNameCol->set(5, "solution status");
//            statValueCol->set(5, "Convergence not reached after maximum number of iterations");
//        }
//        else
//        {
//            statNameCol->set(5, "solution status");
//            statValueCol->set(5, "Converged");
//        }
//    }
//
//}
//
//void PALLogisticRegression_new::loadLabel(Eigen::VectorXd& label)
//{
//    // The two variables are used for mappling the string type
//    // such as YES-----1/NO-----0
//    ltt_adp::string classmap0, classmap1;
//
//    try{
//        getStringArgValue(classmap0, "CLASS_MAP0", iargsTab);
//    }
//    catch (const ltt::exception &){
//        // Do nothing
//    }
//
//    try{
//        getStringArgValue(classmap1, "CLASS_MAP1", iargsTab);
//    }
//    catch (const ltt::exception &){
//        // Do nothing
//    }
//
//    // get label data
//    uint32_t rowNumber = dataTab->getColumn(0)->size();
//    int32_t tmpValue = 0;
//    uint32_t YPos = dataTab->getNumColumns() - 1;
//    ltt_adp::string strValue;
//
//    TRexEnums::AttributeType::typeenum dataType =
//        dataTab->getColumn(YPos)->getValueType();
//
//    if (dataType != TRexEnums::AttributeType::INT &&
//            dataType == TRexEnums::AttributeType::STRING){
//        for (uint32_t row = 0; row < rowNumber; ++row){
//            dataTab->getColumn(YPos)->get(row, strValue);
//
//            if (classmap0 == strValue){
//                if(LR_Args.method == 6)
//                    tmpValue = -1;
//                else
//                    tmpValue = 0;
//            }
//            else if (classmap1 == strValue){
//                tmpValue = 1;
//            }
//            else{
//                THROW_PAL_ERROR(ERR_PAL_CLASSIFY_INVALID_VALUE,
//                        ltt::msgarg_text("APPEND",
//                            "The value of CLASS_MAP0 or CLASS_MAP1 is incorrect."));
//            }
//            label[row] = tmpValue;
//        }
//    }
//    else{
//        for (uint32_t row = 0; row < rowNumber; ++row){
//            dataTab->getColumn(YPos)->get(row, tmpValue);
//
//            if ((tmpValue != 0) && (tmpValue != 1)){
//                THROW_PAL_ERROR(ERR_PAL_CLASSIFY_INVALID_VALUE,
//                        ltt::msgarg_text("APPEND",
//                            "The value of type should be 0 or 1."));
//            }
//            if(tmpValue == 0 && LR_Args.method == 6)
//                tmpValue = -1;
//            label[row] = tmpValue;
//        }
//    }
//
//}
//
//void PALLogisticRegression_new::checkAndValidateTables()
//{
//
//    // check the validity of the input data
//    if (dataTab->getColumn(0)->size() == 0)
//    {
//        THROW_PAL_ERROR(ERR_PAL_COMM_TABLE_EMPTY,
//                ltt::msgarg_text("TAB", dataTab->getName()));
//    }
//
//    if (dataTab->hasNulls())
//    {
//        THROW_PAL_ERROR(ERR_PAL_COMM_NULL_VALUES_IN_TABLE,
//                ltt::msgarg_text("TAB", dataTab->getName()));
//    }
//
//    const uint32_t rowNumber = dataTab->getColumn(0)->size();
//
//    if (rowNumber <= 1)
//    {
//        THROW_PAL_ERROR(ERR_PAL_CLASSIFY_SHORT_DATA,
//                ltt::msgarg_int("COE_NUM", LR_Args.betaNumber + 1));
//    }
//
//    if (static_cast<int32_t>(rowNumber) <= LR_Args.betaNumber)
//    {
//        THROW_PAL_ERROR(ERR_PAL_CLASSIFY_SHORT_DATA,
//                ltt::msgarg_int("COE_NUM", LR_Args.betaNumber + 1));
//    }
//
//    // When there is pvalue output
//    // additional two columns are included
//    if (1 == LR_Args.pvalue && 4 != coefficientTab->getNumColumns())
//    {
//        THROW_PAL_ERROR(ERR_PAL_COMM_TABLE_COL_SIZE,
//                ltt::msgarg_int("COL_SIZE", coefficientTab->getNumColumns())
//                        << ltt::msgarg_text("TAB", coefficientTab->getName())
//                        << ltt::msgarg_int("EXP_COL_SIZE", 4));
//    }
//
//    // This part is for feature selection validaiton
//    // SELECTED_FEATURES and VARIABLE_NUM couldn't be specified together
//    if (LR_Args.featureStr != "")
//    {
//        if (iargsTab->getColumn(0)->findNext(0, _STL::string("VARIABLE_NUM")) < iargsTab->getColumn(0)->size())
//        {
//            THROW_PAL_ERROR(ERR_PAL_INVALID_PARAMETER,
//                    ltt::msgarg_text("MSG",
//                        "SELECTED_FEATURES and VARIABLE_NUM shouldn't be used at the same time"));
//        }
//    }
//
//    // check whether the coefficient should be identified by key or ID
//    TRexEnums::AttributeType idType =
//        coefficientTab->getColumn(0)->getTypeDefinition().getAttributeType();
//
//    if (idType == TRexEnums::AttributeType::STRING ||
//            idType == TRexEnums::AttributeType::CLOB ||
//            idType == TRexEnums::AttributeType::NCLOB)
//    {
//        outputCoeByKey = true;
//    }
//    else
//    {
//        outputCoeByKey = false;
//    }
//
//    // If SELECTED_FEATURES or DEPENDENT_VARIABLE is specified,
//    // the coefficients in the output table could only be identified by key
//    if ((LR_Args.depVar != "" || LR_Args.featureStr != "") &&
//            outputCoeByKey == false)
//    {
//        THROW_PAL_ERROR(ERR_PAL_INVALID_COLUMN_TYPE,
//                ltt::msgarg_text("VAL", coefficientTab->getColumn(0)->getName() + " is not a string type"));
//    }
//
//
//}
//
//template<typename T>
//void PALLogisticRegression_new::outputVector(ltt_adp::vector<T> vec)
//{
//    ltt::cout<<"print element of vector..."<<ltt::endl;
//    size_t vecSize = vec.size();
//    for(size_t i=0; i<vecSize; ++i){
//        ltt::cout<< vec[i] << "\t";
//        if((i+1)%10==0)
//            ltt::cout<<ltt::endl;
//    }
//    ltt::cout<<"\nprint vector finished..."<<ltt::endl;
//
//}

}// end of LOGR

}// end of AFL_PAL



