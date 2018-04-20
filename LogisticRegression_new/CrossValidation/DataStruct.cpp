#include "AFL/PAL/Classification/LogisticRegression_new/CrossValidation/DataStruct.h"
#include "AFL/PAL/Utilities/Matrix/Matrix.hpp"
#include <iostream>
#include "AFL/PAL/Utilities/EigenCommon.h"

USE_ERROR(AFL_PAL, ERR_PAL_CLASSIFY_INVALID_VALUE);

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
    // get label data
    int32_t tmpVal = 0;
    ltt_adp::string strVal;
    uint32_t YPos = dataView->getNumColumns() - 1;
    uint32_t nrow = dataView->getColumn(0).size();

    m_yValue.resize(nrow);
    ltt_adp::string classmap0, classmap1;
    if(classMap.size() > 0){
        if(classMap.size() != 2){
            THROW_PAL_ERROR(ERR_PAL_CLASSIFY_INVALID_VALUE, ltt::msgarg_text("APPEND",
                            "The value of CLASS_MAP0 or CLASS_MAP1 does not exist."));
        }
        classmap0 = classMap[0];
        classmap1 = classMap[1];
    }

    TRexEnums::AttributeType::typeenum dataType = dataView->getColumn(YPos).getValueType();
    if(dataType != TRexEnums::AttributeType::INT && dataType == TRexEnums::AttributeType::STRING){
        for(uint32_t row=0; row<nrow; ++row){
            dataView->getColumn(YPos).get(row, strVal);
            if(classmap0 == strVal){
                if(method == PGD) tmpVal = -1;
                else tmpVal = 0;
            }
            else if(classmap1 == strVal) tmpVal = 1;
            else{
                THROW_PAL_ERROR(ERR_PAL_CLASSIFY_INVALID_VALUE, ltt::msgarg_text("APPEND",
                                  "The value of CLASS_MAP0 or CLASS_MAP1 is incorrect."));
            }
            m_yValue(row) = tmpVal;
        }
    }
    else{
        for(uint32_t row=0; row<nrow; ++row){
            dataView->getColumn(YPos).get(row, tmpVal);

            if(tmpVal !=0 && tmpVal !=1)
            {
                THROW_PAL_ERROR(ERR_PAL_CLASSIFY_INVALID_VALUE, ltt::msgarg_text("APPEND",
                                  "The value of CLASS_MAP0 or CLASS_MAP1 is incorrect."));
            }
            if(tmpVal == 0 && method == PGD) tmpVal = -1;
            m_yValue(row) = tmpVal;
        }
    }
}

void LogRTrainData::loadFeatureData(Utility::DataTableViewBase const* dataView,
                              ltt_adp::vector<bool>& isCategoryVec, bool& isCV)
{
    uint32_t continuousColNum = 0;
    for(size_t i=0; i<isCategoryVec.size(); ++i){// count number of the continuous column
        if(!isCategoryVec[i]) ++continuousColNum;
    }
    m_continuousColNum = continuousColNum + 1; // add 1 because of intercept term

    _STL::vector<T> tripletList;
    uint32_t ncol = dataView->getNumColumns();
    uint32_t nrow = dataView->getColumn(0).size();
    m_categoryColNum = ncol - m_continuousColNum;

    m_dsMatrix.denseMat.resize(nrow, continuousColNum+1);// add 1 because of intercept term
    m_dsMatrix.denseMat.col(0).setOnes(); // intercept term

    uint32_t expandCategoryNum = 0;
    uint32_t cateInd = 0;
    uint32_t colIndex = 0;
    uint32_t catNum = 0;
    uint32_t contNum = 0;
    //uint32_t hasCateThanOneIdx=0;
    for(uint32_t colInd=0; colInd<ncol-1; ++colInd){// last column is label
        if(!isCategoryVec[colInd]){// continuous column
            m_categoryNumOfEachColumn.push_back(0);
            ++contNum;
            for(uint32_t rowInd=0; rowInd<nrow; ++rowInd){
                dataView->get(colInd, rowInd, m_dsMatrix.denseMat(rowInd, contNum));
            }
            ++colIndex;
        }
        else{// category column
            //++hasCateThanOneIdx;
            Utility::DataColumnView categoryColView = dataView->getColumn(colInd);
            Utility::DataColumnView const* categoryCol = &categoryColView;
            uint32_t dictSize = 0;
            if(isCV){// for cross validation
                handleCategoryColumn(categoryCol, nrow, tripletList, cateInd, dictSize);
                expandCategoryNum += dictSize-1;
                m_categoryNumOfEachColumn.push_back(dictSize);
                ++catNum;
                m_insPos.push_back(colIndex+catNum);
                colIndex += dictSize-1;
                //if(dictSize > 1) m_hasCateThanOneIdx.push_back(hasCateThanOneIdx);
            }
            else{// normal training
                const TRexCommonObjects::ColumnDictBase* dict = dataView->getColumn(colInd).getDict();
                uint32_t dictSize = dict->size(); //XXX: the size may be not the number of distinct values.
                uint32_t numDistinct = 0;
                _STL::vector<_STL::string> sortedDict(*m_alloc);
                for(uint32_t i=0; i<dictSize; ++i){
                    _STL::string tmpStr;
                    if(dict->get(i, tmpStr)){
                        ++numDistinct;
                        sortedDict.emplace_back(tmpStr);
                    }
                }
                m_categoryNumOfEachColumn.push_back(numDistinct);
                expandCategoryNum += numDistinct-1;
                ltt::sort(sortedDict.begin(), sortedDict.end(), *m_alloc);
                for(uint32_t i=1; i<numDistinct; ++i){ // start from 1, remove the first category
                    ltt_adp::string tmpStr;
                    tmpStr = sortedDict[i];
                    ltt_adp::vector<uint32_t> indexVec;
                    dict->find(tmpStr, indexVec);
                    for(uint32_t ii=0; ii<indexVec.size(); ++ii){
                        tripletList.push_back(T(indexVec[ii],cateInd, 1));
                    }
                    ++cateInd;
                }
                ++catNum;
                m_insPos.push_back(colIndex+catNum);
                colIndex += numDistinct - 1;
            }// end of normal training
        }// end category column

    }// end of colIndex
    if(expandCategoryNum > 0){
        m_dsMatrix.sparseMat.resize(nrow, expandCategoryNum);
        m_dsMatrix.sparseMat.setFromTriplets(tripletList.begin(), tripletList.end());
    }

}

void LogRTrainData::handleCategoryColumn(Utility::DataColumnView const* categoryCol, uint32_t& nrow,
                           _STL::vector<T>& tripletList, uint32_t& cateInd, uint32_t& dictSize)
{
    ltt_adp::map<_STL::string, _STL::vector<uint32_t>> cateMap;
    ltt_adp::map<_STL::string, _STL::vector<uint32_t>>::iterator it;

    for(uint32_t rowIdx=0; rowIdx<nrow; ++rowIdx){// find different categorical value and record the location in row
        ltt_adp::string strVal;
        categoryCol->get(rowIdx, strVal);
        it = cateMap.find(strVal);
        if(it == cateMap.end()){
            _STL::vector<uint32_t> rowIdxVec(*m_alloc);
            rowIdxVec.push_back(rowIdx);
            cateMap[strVal] = rowIdxVec;
        }
        else{
            _STL::vector<uint32_t>& rowIdxVec = it->second;
            rowIdxVec.push_back(rowIdx);
            cateMap[strVal] = rowIdxVec;
        }
    }

    dictSize = cateMap.size();
    it = cateMap.begin();
    _STL::vector<_STL::string> cateVec(*m_alloc);
    for(uint32_t i=1; i<dictSize; ++i){ // one-hot encoding, remove first categorical value
        ++it;
        cateVec.push_back(it->first);
        _STL::vector<uint32_t> rowIdxVec = it->second;
        for(uint32_t ii=0; ii<rowIdxVec.size(); ++i){
            tripletList.push_back(T(rowIdxVec[ii], cateInd, 1));
        }
        ++cateInd;
    }
    m_categoryEachColumn.push_back(cateVec);

}

void LogRTrainData::standardize()
{
    m_dsMatrix.standardize(m_meanVec, m_stdVec);
}


/*********   load predict data   *********/
void LogRPredictData::loadData(Utility::DataTableViewBase const* pDataView,
                               ltt_adp::vector<bool>& isCategoryVec)
{
    _STL::vector<T> tripletList;

    uint32_t continuousColNum = 0;
    for(size_t i=0; i<isCategoryVec.size(); ++i){
        if(!isCategoryVec[i]) ++continuousColNum;
    }
    uint32_t ncol = pDataView->getNumColumns();
    uint32_t nrow = pDataView->getColumn(0).size();
    m_continuousColNum = continuousColNum + 1; // add 1 because of intercept term
    m_dsMatrix.denseMat.resize(nrow, continuousColNum+1);// add 1 because of intercept term
    m_dsMatrix.denseMat.col(0).setOnes();// intercept term

    uint32_t cateIdx=0;
    uint32_t contIdx=0;
    uint32_t totalCategoryCols = 0;
    uint32_t dictSize = 0;
    for(uint32_t colIdx=0; colIdx<ncol; ++colIdx){
        if(!isCategoryVec[colIdx]){// continuous column
            for(uint32_t rowIdx=0; rowIdx<nrow; ++rowIdx){
                pDataView->get(colIdx, rowIdx, m_dsMatrix.denseMat(rowIdx, contIdx+1));
            }
            ++contIdx;
        }
        else{// category column
            ltt::vector<_STL::string> cateEachColumn(*m_alloc);
            _STL::string strVal;
            ltt_adp::map<_STL::string, ltt_adp::vector<uint32_t>> cateMap;
            ltt_adp::map<_STL::string, ltt_adp::vector<uint32_t>>::iterator it;

            for(uint32_t rowIdx=0; rowIdx<nrow; ++rowIdx){
                pDataView->get(colIdx, rowIdx, strVal);
                //cateEachColumn.push_back(strVal);

                it = cateMap.find(strVal);
                if(it == cateMap.end()){
                    ltt_adp::vector<uint32_t> rowIdxVec;
                    rowIdxVec.push_back(rowIdx);
                    cateMap[strVal] = rowIdxVec;
                }
                else{
                    ltt_adp::vector<uint32_t> rowIdxVec = it->second;
                    rowIdxVec.push_back(rowIdx);
                    cateMap[strVal] = rowIdxVec;
                }

            }
            eachColumnCategoryIdx.push_back(cateMap);

            it = cateMap.begin();
            dictSize = cateMap.size();
            _STL::vector<_STL::string> cateVec;
            for(uint32_t i=0; i<dictSize; ++i){
                cateVec.push_back(it->first);
                _STL::vector<uint32_t> rowIdxVec = it->second;
                for(uint32_t ii=0; ii<rowIdxVec.size(); ++ii){
                    tripletList.push_back(T(rowIdxVec[ii], cateIdx, 1));
                }
                ++cateIdx;
            }
            categoryVec.push_back(cateVec);
        }// end of category column
        totalCategoryCols += dictSize;

    }// end of for(uint32_t colIdx...)

    if(totalCategoryCols > 0){
        m_dsMatrix.sparseMat.resize(nrow, totalCategoryCols);
        m_dsMatrix.sparseMat.setFromTriplets(tripletList.begin(), tripletList.end());
    }

}


/*********   load compare data   *********/
void LogRCompareData::loadData(Utility::DataTableViewBase const* dataView,
                            ltt::vector<_STL::string>& classMap)
{
    // get label data
    int32_t tmpVal = 0;
    ltt_adp::string strVal;
    uint32_t YPos = dataView->getNumColumns() - 1;
    uint32_t nrow = dataView->getColumn(0).size();
    compareData.resize(nrow);

    ltt_adp::string classmap0, classmap1;
    if(classMap.size() > 0){
        if(classMap.size() != 2){
            THROW_PAL_ERROR(ERR_PAL_CLASSIFY_INVALID_VALUE,                                                        
                    ltt::msgarg_text("APPEND",
                        "The value of CLASS_MAP0 or CLASS_MAP1 not exist."));
        }
        classmap0 = classMap[0];
        classmap1 = classMap[1];
    }
    TRexEnums::AttributeType::typeenum dataType = dataView->getColumn(YPos).getValueType();
    if(dataType != TRexEnums::AttributeType::INT && dataType == TRexEnums::AttributeType::STRING){
        for(uint32_t row=0; row<nrow; ++row){
            dataView->getColumn(YPos).get(row, strVal);
            m_stringLabels.push_back(strVal);

            if(classmap0 == strVal){
                tmpVal = 0;
            }
            else if(classmap1 == strVal){
                tmpVal = 1;
            }
            else{
                THROW_PAL_ERROR(ERR_PAL_CLASSIFY_INVALID_VALUE,
                        ltt::msgarg_text("APPEND",
                            "The value of CLASS_MAP0 or CLASS_MAP1 is incorrect."));
            }
            compareData(row) = tmpVal;
        }
    }
    else{
        for(uint32_t row=0; row<nrow; ++row){
            dataView->getColumn(YPos).get(row, tmpVal);
            ltt_adp::stringstream ss(ltt::ios_base::goodbit);

            if(tmpVal !=0 && tmpVal !=1){
                THROW_PAL_ERROR(ERR_PAL_CLASSIFY_INVALID_VALUE,
                        ltt::msgarg_text("APPEND",
                            "The value of CLASS_MAP0 or CLASS_MAP1 is incorrect."));
            }
            compareData(row) = tmpVal;
            ss << tmpVal;
            m_stringLabels.push_back(ss.str());
        }
    }

}

} // end of LogR
} // end of AFL_PAL





