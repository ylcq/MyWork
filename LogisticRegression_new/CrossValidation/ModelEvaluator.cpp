#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/ModelEvaluator.h"
#include "AFL/PAL/Utilities/CrossValidation/EvaluateMeasure.h"
#include "AFL/PAL/ModelEvaluation/Utilities/PALModelEvaluateUtility.h"
#include "AFL/PAL/ModelEvaluation/AUC/AUC.h"
#include "AFL/PAL/ModelEvaluation/AUC/palAUC.h"

USE_ERROR(AFL_PAL,ERR_PAL_MODELEVALUATION_WRONG_INPUT);

namespace AFL_PAL
{
namespace LOGR
{
double LogREvaluator::operator()(
            LogRModel const* pModel,
            LogRPredictData const* pPredictData,
            LogRCompareData const* pCompareData,
            Args const* pArgs,
            size_t threadNum) const
{
    int metric_type = pArgs->LR_Args.metric_type;
    double value = 0.0;

    _STL::vector<double>& betaValue_ = (const_cast<LogRModel*>(pModel))->getRetBackCoefficient();
    size_t nTrainContinuousCol = (const_cast<LogRModel*>(pModel))->getContinuousColNum();
    size_t nTrainCoeffSize = betaValue_.size();

    _STL::vector<_STL::vector<_STL::string>>& trainCategoryVec = (const_cast<LogRModel*>(pModel))->getCategoryEachColumn();

    dense_sparse_matrix::DSMatrix& predictData = (const_cast<LogRPredictData*>(pPredictData))->getPredictData();

    _STL::vector<_STL::vector<_STL::string>>& predictCategoryVec = (const_cast<LogRPredictData*>(pPredictData))->getCategoryVec();

    size_t nPredictCol = predictData.cols();
    size_t nPredictContinuousCol = predictData.denseMat.cols();
    size_t nPredictCategoryCol = predictData.sparseMat.cols();
    Eigen::VectorXd betaValue(nPredictCol);

    for(size_t i=0; i<nPredictContinuousCol; ++i){// continuous coefficient
        betaValue(i) = betaValue_[i];
    }

    size_t predictPreviousCategoryNum = 0;
    size_t trainPreviousCategoryNum = 0;
    for(size_t i=0; i<predictCategoryVec.size(); ++i){// category coefficient // loop for column
        for(size_t j=0; j<predictCategoryVec[i].size(); ++j){// each column of predict column
            _STL::string categoryName = predictCategoryVec[i][j];
            int idx = -1;
            for(size_t jj=0; jj<trainCategoryVec[i].size(); ++jj){// find category name in train model
                ++idx;
                if(categoryName == trainCategoryVec[i][jj]){
                    break;
                }
            }// end of for(trainCategoryVec)
            if(idx <= -1){// category does not occur in train model
                betaValue(nPredictContinuousCol + j + predictPreviousCategoryNum) = 0;
            }
            else{// category occur in train model
                betaValue(nPredictContinuousCol + j + predictPreviousCategoryNum) = betaValue_[nPredictContinuousCol + idx + trainPreviousCategoryNum];
            }
        }// end of each column of predict column
        predictPreviousCategoryNum = predictCategoryVec[i].size();
        trainPreviousCategoryNum = trainCategoryVec[i].size();
    }// end of loop in column

    size_t nrow = predictData.rows();
    _STL::vector<_STL::string>& cmpStrLabels = (const_cast<LogRCompareData*>(pCompareData))->getStrLabels();
    _STL::vector<_STL::string> predStrLabels;
    if(cmpStrLabels.size() != nrow){
        THROW_PAL_ERROR(ERR_PAL_MODELEVALUATION_WRONG_INPUT,
             ltt::msgarg_text("APPEND", "the size of the data table and result table are not equal or empty"));
    }

    Eigen::VectorXd& compareData = (const_cast<LogRCompareData*>(pCompareData))->getCompareData();
    predStrLabels.resize(nrow);
    _STL::string classmap0;
    _STL::string classmap1;
    bool isStrLabel = pArgs->classMap.size() > 0 ? true: false;
    if(isStrLabel){
        classmap0 = pArgs->classMap[0];
        classmap1 = pArgs->classMap[1];
    }
    Eigen::VectorXd result(nrow);
    Eigen::VectorXd prob(nrow);
    dense_sparse_matrix::DSMatrix::matrixVectorProduct<Eigen::VectorXd, Eigen::VectorXd>(predictData, betaValue, result, m_alloc, threadNum);

    // accuracy
    int trueClassify = 0;
    double accuracy = 0.0;
    double lossSum = 0.0; // log loss

    if(isStrLabel){
        for(size_t i=0; i<nrow; ++i){
            if(result(i) > 0){
                prob(i) = 1.0/(1+exp(-result(i)));
                predStrLabels[i] = classmap1;
                if(compareData(i) == 1) ++trueClassify;
            }
            else{
                prob(i) = exp(result(i)) / (1+exp(result(i)));
                predStrLabels[i] = classmap0;
                if(compareData(i) == 0) ++trueClassify;
            }
            double log_p = prob(i) > 1e-10 ? log(prob(i)) : 0;
            double log_1p = (1-prob(i)) > 1e-10 ? log(1-prob(i)) : 0;
            lossSum += compareData(i) * log_p + (1-compareData(i)) * log_1p;
        }
    }
    else{
        for(size_t i=0; i<nrow; ++i){
            if(result(i) > 0){
                prob(i) = 1.0/(1+exp(-result(i)));
                ltt_adp::stringstream ss(ltt::ios_base::goodbit);
                ss << 1;
                _STL::string stmp = ss.str();
                predStrLabels[i] = stmp;
                if(compareData(i) == 1) ++trueClassify;
            }
            else{
                prob(i) = exp(result(i)) / (1+exp(result(i)));
                ltt_adp::stringstream ss(ltt::ios_base::goodbit);
                ss << 0;
                _STL::string stmp = ss.str();
                predStrLabels[i] = stmp;
                if(compareData(i) == 0) ++trueClassify;
            }
            double log_p = prob(i) > 1e-10 ? log(prob(i)) : 0;
            double log_1p = (1-prob(i)) > 1e-10 ? log(1-prob(i)) : 0;
            lossSum += compareData(i) * log_p + (1-compareData(i)) * log_1p;
        }
    }
    accuracy = (1.0*trueClassify)/nrow;
    if(metric_type == ACCURACY) value = accuracy;
    if(metric_type == NLL) value = -1*lossSum;

    // F1-score
    double f1 = 0.0;
    if(metric_type == F1_SCORE){
        CV::F1ScoreMeasure<_STL::string const*, _STL::string const*> measureF1;
        f1 = measureF1(cmpStrLabels.data(), predStrLabels.data(), nrow);
    }

    // roc_auc
    double roc_auc = 0.0;
    if(metric_type == AUC){
        _STL::vector<LabelProb> label_prob(nrow);
        for(size_t i=0; i<nrow; ++i)
        {
            label_prob[i].originalLabel = compareData(i);
            label_prob[i].Prob = prob(i);
        }
        AFL_PAL::PALAUCPop pop;
        roc_auc = pop.getAUC(1, label_prob);
    }

    if(metric_type == F1_SCORE) value = f1;
    if(metric_type == AUC) value = roc_auc;

    return value;
}


}// end of LOGR
}// end of AFL_PAL


