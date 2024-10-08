---
title: BB程式說明

---

## 檔案說明

1. BB_EEE
    :    Branch and Bound的函數都放在這一個檔案裏面
2. MuMIMOClass
    :    我們的通訊環境是基於這一個函式庫去建立的
3. Train_main
    :    主程式，我們要執行的就是這一個檔案
4. param.json
    :    用來設定實驗參數用的json檔

## 執行方法

1. 首先我們要先進入到一個裝有BB所使用的求解器的環境，安裝流程請見[BB求解器安裝說明](bonmin_readme.txt)，並且進入到該求解器環境中
2. 到[param.json](./src/BB/param.json)當中調整實驗用的參數，這邊我列出可以被你調整的參數
    *    NumRISEle
            :   RIS反射元件的數量
    *    NumUE
            :   用戶的數量
    *    Bandwidth
            :   通訊環境的頻寬
    *    QoS_exponent
            :   延遲服務品質指數，用於提整QoS嚴格程度，詳情請見論文
    *    LocalDataSize
            :   用戶傳輸的資料量
    *    Position
            :   調整用戶與BS以及RIS的位置
    *    BB/EPISODES
            :   進行多少次傳輸後才計算平均EEE(Effective Energy Efficiency)
    *    UserActionSpace
            :   用戶可選擇的傳輸功率有幾個
    *    RISActionSpace
            :   RIS可選擇的相位偏移角度有幾個
    *    K_U2B、K_R2B、K_U2R
            :   Rician K factor

3. (optional)可以使用[src](./src)底下的[datasets_generation](./src/datasets_generation/)底下的[calculate_recommended_data_size.py](./src/datasets_generation/calculate_recommended_data_size.py)來確認自己設定的參數是否可行(看資料量大小是否傳得完)

**注意! 這個步驟要在docker外面完成**

```python!
cd ../datasets_generation
datasets_generation$ python3 calculate_recommended_data_size.py
```

4. 設定好實驗參數後就可以回到環境中執行程式了


```python!
docker attach <docker container名稱>
cd ~/BB
~/BB$ python3 Train_main.py
```
