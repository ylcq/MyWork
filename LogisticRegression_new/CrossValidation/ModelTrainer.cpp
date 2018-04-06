#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/ModelTrainer.h"

namespace AFL_PAL
{
namespace LOGR
{
_STL::unique_ptr<LogRModel> LogRModelTrainer::operator()(
                 LogRTrainData const* pData,
                 TRexCommonObjects::InternalTable const* paramTBL,
                 Args const* pArgs,
                 size_t threadNum,
                 double* pErr,
                 _STL::vector<double>* pErrlog) const
{
    PalParameters<double> ppd(const_cast<TRexCommonObjects::InternalTable*> (paramTBL));
    ppd.registerParameter("ENET_LAMBDA", 0.0, new DoubleBtCheck(0.0, true));
    ppd.registerParameter("ENET_ALPHA", 1.0, new DoubleRangeCheck(0, 1, true, true));
    double lambda = ppd["ENET_LAMBDA"];
    double alpha = ppd["ENET_ALPHA"];
    pArgs->LR_Args.lambda = lambda;
    pArgs->LR_Args.alpha = alpha;

    _STL::unique_ptr<LogRModel> pModel(new(*m_alloc) LogRModel(*m_alloc));

    dense_sparse_matrix::DSMatrix& xValue = (const_cast<LogRTrainData*> (pData))->getDSMatrix();
    Eigen::VectorXd& yValue = (const_cast<LogRTrainData*> (pData))->getYValue();
    ltt_adp::vector<double>& meanVec = (const_cast<LogRTrainData*> (pData))->getMeanVec();
    ltt_adp::vector<double>& stdVec = (const_cast<LogRTrainData*> (pData))->getStdVec();
    uint32_t continuousColNum = (const_cast<LogRTrainData*> (pData))->getContinuousColNum();
    uint32_t categoryColNum = (const_cast<LogRTrainData*> (pData))->getCategoryColNum();
    _STL::vector<_STL::vector<_STL::string>>& categoryEachColumn = (const_cast<LogRTrainData*> (pData))->getCategoryEachColumn();

    pModel->setCategoryEachColumn(categoryEachColumn);// unique categorical value of category column
    pModel->setContinuousColNum(continuousColNum);// number of continuous column
    pModel->setCategoryColNum(categoryColNum);// number of category column

    pModel->modelTrain(xValue, yValue, pArgs->LR_Args, pArgs->argsTab, lambda, alpha, threadNum);

    size_t feature_dimension = pModel->getFeatureDimension();
    size_t expandColSize = pModel->getExpandColSize();
    _STL::vector<double>& betaValue = pModel->getCoefficient();
    _STL::vector<double>& retBackCoeff = pModel->getRetBackCoefficient();
    _STL::vector<double>& nretBackCoeff = pModel->getNoRetBackCoefficient();

    retBackCoeff.resize(feature_dimension);
    nretBackCoeff.resize(feature_dimension);

    int method = pArgs->LR_Args.method;
    bool isStandardize = pArgs->LR_Args.standardize==1;
    if(method != CCD){
        double tmpVal = 0.0;
        for(size_t i=0; i<betaValue.size(); ++i){
            nretBackCoeff[i] = betaValue[i];
            if(isStandardize){
                if(stdVec[i] > 0){
                    tmpVal += betaValue[i] * meanVec[i]/stdVec[i];
                    retBackCoeff[i] = betaValue[i]/stdVec[i];
                }
            }
            else retBackCoeff[i] = betaValue[i];
        }
        if(isStandardize) retBackCoeff[0] = betaValue[0] - tmpVal;
        else retBackCoeff[0] = betaValue[0];
    }
    return pModel;

}



}// end of LOGR
}// end of AFL_PAL


