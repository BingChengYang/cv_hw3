# cv_hw3

* 1.Training
* 2.Testing


## Training model
使用以下指令訓練模型:
```
python train.py --lr=0.0001 --epoches=10000 --mini_batch_size=32
```
lr代表learning rate的大小，default = 0.0001</br>
epoches代表總共訓練幾個epoch，default = 10000</br>
mini_batch_size代表會使用mini_batch的大小，default = 32</br>

## Testing model
使用下列指令來產生預測結果:
```
python test.py --test_model="model_final.pkl"
```
test_model代表所要選擇測試的model，default = "model_final.pkl"</br>
結束後會在submit資料夾產生0756545.json
