# 程式說明閱讀順序與環境安裝

## 安裝環境

環境所需檔案皆已包裝在[requirements.sh](./requirements.sh)當中

```shell!
sh ./requirements.sh
```

## 通訊環境建構

本研究使用之RIS通訊環境來自這一個[repo](https://github.com/WeiWang-WYS/IRSconfigurationDRL)

## Branch and Bound

如果要使用Branch and Bound 程式要看[BB程式使用說明](BB程式說明.md)

## Decision Transformer

如果要使用Decision Transformer 程式 需要先了解**資料集生成用程式**，之後才可以開始使用Decision Transformer

##### 資料集生成說明

要了解資料集生成用程式，請看[生成資料集用程式說明](生成資料集用程式說明.md)

在資料集生成之後，記得要將資料集移到Decision Transformer的workspace底下才可以讓程式成功讀取

```shell!
datasets_generation$ cp <datasets名稱>.npz <DT workspace 路徑>
Ex: cp User_10_with_8_RIS_complete_datasets.npz ../DT_experiment
```

##### Decision Transformer使用說明

請看[Decisoin Transformer程式說明](DT程式說明.md)
