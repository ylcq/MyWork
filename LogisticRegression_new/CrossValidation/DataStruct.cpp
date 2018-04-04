#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataStruct.h"

namespace AFL_PAL
{
namespace LOGR
{

/*********   load train data    *********/
void LogRTrainData::loadData(Utility::DataTableViewBase const* pDataView,
                             ltt::vector<_STL::string>& classMap,
                             ltt_adp::vector<bool>& isCategoryVec, int& method, bool& isCV)
{
    loadLabelData(pDataView, classMap, method);

    loadFeatureData(pDataView, isCategoryVec, isCV);

}

void LogRTrainData::loadLabelData(Utility::DataTableViewBase const* dataView,
                              ltt::vector<_STL::string>& classMap, int& method)
{


}

void LogRTrainData::loadFeatureData(Utility::DataTableViewBase const* dataView,
                              ltt_adp::vector<bool>& isCategoryVec, bool& isCV)
{
    uint32_t continuousColNum = 0;
    for(size_t i=0; i<isCategoryVec.size(); ++i){// count number of the continuous column
        if(!isCategoryVec[i]) ++continuousColNum;
    }
    _STL::vector<T> tripletList;
    uint32_t ncol = dataView->getNumColumns();
    uint32_t nrow = dataView->getColumn(0).size();
    dsMatrix.denseMat.resize(nrow, continuousColNum+1);// add 1 because of intercept term
    dsMatrix.denseMat.col(0).setOnes(); // intercept term

    uint32_t totalCategoryCols = 0;
    uint32_t cateInd = 0;
    uint32_t colIndex = 0;
    uint32_t catNum = 0;
    uint32_t contNum = 0;
    for(uint32_t colInd=0; colInd<ncol-1; ++colInd){// last column is label
        if(!isCategoryVec[colInd]){// continuous column
            categoryNumOfEachColumn.push_back(0);
            ++contNum;
            for(uint32_t rowInd=0; rowInd<nrow; ++rowInd){
                dataView->get(colInd, rowInd, dsMatrix.denseMat(rowInd, contNum));
            }
            ++colIndex;
        }
        else{// category column
            Utility::DataColumnView categoryColView = dataView->getColumn(colInd);
            Utility::DataColumnView const* categoryCol = &categoryColView;
            uint32_t dictSize = 0;
            if(isCV)
                handleCategoryColumn(categoryCol, nrow, tripletList, cateInd, dictSize);
        }

    }

}

void LogRTrainData::handleCategoryColumn(Utility::DataColumnView const* categoryCol, uint32_t& nrow,
                           _STL::vector<T>& tripletList, uint32_t& cateInd, uint32_t& dictSize)
{
    for(uint32_t rowIdx=0; rowIdx<nrow; ++rowIdx){
        ltt_adp::string strVal;

    }

}

/*********   load predict data   *********/
void LogRPredictData::loadData(Utility::DataTableViewBase const* pDataView,
                               ltt_adp::vector<bool>& isCategoryVec)
{


}


/*********   load compare data   *********/
void LogRCompareData::loadData(Utility::DataTableViewBase const* pDataView,
                            ltt::vector<_STL::string>& classMap)
{


}

} // end of LogR
} // end of AFL_PAL





