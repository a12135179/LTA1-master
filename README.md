LaneDetection Project 
===
本專案基於下列兩篇source code做修改 
1.https://github.com/ZJULearning/resa.git
2.https://github.com/hustvl/YOLOP.git
### 1, 環境建置
---
這個專案主要用pytorch實作，
首先，建置虛擬環境：
```
conda create --name LaneDetect python=3.7
activate LaneDetect
```
再來下載相關套件

```setup
pip install -r requirements.txt
```
___
### ２, Training
-----
#### 事前準備
訓練資料必須以 Tusimple Dataset的標註方式做標註。資料夾裡面會包含一個資料夾跟一個json檔案：gt_image 跟 JsonFile_Label.json。看看專案資料夾底下有沒有weight資料夾
，如果沒有這個資料夾就創一個

    your_data_folder 
        |-gt_image/ 
        |-JsonFile_label.json
檢查一下json檔，裡面不能有車道線0條的標註，也就是
{"lanes":[]} 或是 {}
#### Step 1
有了標註完的資料集Dataset(標註的資料要放在一個資料夾底下)，為了方便管理訓練資料，將資料夾放進  ./data 底下，如果沒有這個資料夾就創一個，並另外創立兩個json檔案，test_label.json and valset.json。
    
    data/your_data_folder 
        |-gt_image/ 
        |-JsonFile_label.json
        |-test_label.json
        |-valset.json

#### ▲Step 2準備測試資料(optional)
從JsonFile_label.json 裡面剪下幾行貼到這兩個json裡面，作為testing set。在每一輪訓練epoch結束之後，就會跑一遍test。
#### ▲Step 3 產生 labels檔案

ground truth會以圖片的形式來做訓練，輸入以下指令
```
>python tools/generate_seg_tusimple.py --root ./data/[your_data_folder]
```
其中，--root 後面要填入資料集的位址。也就是說[your_data_folder]換成資料集的名字。當然，如果不想把訓練資料放在data底下，那麼--root就整個改掉就好。

這個步驟會在dataset底下產生一個seg_label資料夾，如果這個步驟做完之後，想要更改任何一個json檔的話，需要把這個seg_label資料夾刪掉，並重新輸入上面的指令後才可以進行訓練。

#### ▲Step 4 修改config檔案
1.總共需要修改的東西有兩個：pretrain model 跟 dataset_path

首先，到./lib/config/resa_tusimple.py這個文件，找到 "dataset_path" 和
"test_json_file" 並且將後面的內容分別改成 './data/[training dataset dir]]' and './data/[training dataset dir]/test_label.json' 
其中，[training dataset dir]的部分填入訓練資料的資料夾名稱

再來，如果這不是第一次訓練，有pretrained model的話，到
./lib/config/default.py這個文件底下，找到_C.MODEL.PRETRAINED
這個變數，將後面的內容改為pretrained model的位址。
#### ▲Step 5 training
欲開始訓練，請輸入下列指令：
```
python tools/train.py --view --cpu 0 
```

訓練完成之後，去到./work_dirs 資料夾底下，找到最新創立的訓練紀錄資料夾，該資料夾的名稱會是開始訓練的日期時間(比如20220114_162156代表在2022/01/14的16L21:56秒開始訓練)，這個資料夾底下會有ckpt、output這兩個資料夾。ckpt會裝有訓練完成的best.pth檔案，output則會有測試資料的預測結果(tusimple格式)。
--view 可輸入可不輸入，若有打--view的話，前面提到的訓練紀錄資料夾會再有一個vis資料夾，裡面有測試資料的預測可視化結果。

訓練完成的best.pth檔案，出於方便，也會保存一份在./weights底下，並且以dataset的名稱命名。
如果輸入--validate 如圖
```
python tools/train.py --view --cpu 0 --validate
```
這樣一來他不會訓練，而是會去valset.json裡面抓檔案做validate，並輸出accuracy
#### ▲Tusimple Dataset
如果要使用Tusimple的資料集做training，首先要做一些轉換：
Tusimple的資料標註格式的H_samples在label工具中提到是固定的56個y座標，但其實還有
48個y座標的H_sample格式，從y=240~y=710。但是這個格式不能為我們的訓練所用，所以
必須要轉換成56的。Tusimple的Json檔中有的檔案有包含這種形式的H_sample，直接下去train會產生一些error。

step1 找到Json檔案裡面中有包含上述H_samples格式的json。只要在檔案裡面用ctrl+F搜尋 "h_samples": [240  如果有匹配結果就需要做修改。

step2
假設有上述問題的json檔叫做 Jason.json，為了方便辨識，把他改名為Jason_old.json然後創一個空白的Jason.json，
接著輸入以下指令
```
python tools/Label56_trainsformer.py --path Jason_old.json --edit Jason.json
```
新的Jason.json就會生出來，確認裡面的H_samples都是從 160 開始記錄後，把Jason_old.json從檔案裡面拿掉。這樣這個json檔案就可以拿來訓練了
### ３,Demo資料
____

Demo的資料可以是影片video或是圖片image，demo的功能寫在tools/demo.py這個文件裡面。看看專案資料夾底下有沒有demo資料夾，如果沒有這個資料夾就創一個
先打開anaconda Prompt, cd到本專案資料夾底下，並activate 虛擬環境

#### 1.影片：
需要的東西：待測影片、訓練好的weight

無論有一個或數個影片需要demo，都將所有待測影片放在一個資料夾內，並且將該資料夾放入./demo/source 資料夾底下。然後把資料夾的路徑記下來(應為./demo/source/your_demo_video),同時也記下weight的路徑。

執行demo的時候，輸入下列指令
```
python tools/demo.py --gpu 0 --view --source_type video --demo_data_path [your demo_video path] --weight [your weight path]
```
然後，去到./demo/資料夾下面，會發現一個以代測影片的名稱和今天日期為命名的資料夾被創立，假使影片叫做road.mp4,日期是2022_1_24,那麼資料夾會叫做road.mp4-2022-01-24,輸出影片就放在這裡面

--weight 如果不輸入的話，就會預設為./lib/config/default.py這個文件底下的_C.MODEL.PRETRAINED 


#### 2.圖片：
需要的東西：待測圖片、訓練好的weight

無論有一個或數個圖片需要demo，都將所有待測圖片放在一個資料夾內，並且將該資料夾放入./demo/source 資料夾底下。然後把資料夾的路徑記下來(應為./demo/source/your_demo_video),同時也記下weight的路徑。

執行demo時，輸入下列指令
```
python tools/demo.py --gpu 0 --view --source_type image --demo_data_path [your demo_video path] --weight [your weight path]
```


然後，去到./demo/資料夾下面，會發現一個以代測圖片資料夾的名稱和今天日期為命名的資料夾被創立，假使圖片資料夾叫做road,日期是2022_1_24,那麼資料夾會叫做road-2022-01-24,輸出圖片就放在這裡面

---
### 4 config 檔
本專案總共有兩個config檔，放在lib/config底下，有default.py 跟resa_tusimple.py。default.py的部分主要是規定backbone的，會動到的東西只有_C.MODEL.PRETRAINED 而其他東西用不到。resa_tusimple則是負責decoder的部分已經新加入的resanet的參數。除此之外如果要改訓練的epoch、batch size、訓練資料的路徑都是在這邊做修改。

### 5.輸出功能
如果要增加輸出畫面的東西的話，.\lib\runner\evaluator\tusimple\tusimple.py的
class TuSimple_Demo 的 view()方法，這裡面負責印出車道線，而在這裏面呼叫的 
class DrivingAssistant(lib/Panel/DrivingAssist.py) 就是負責其他畫面功能的追加。如果要增加東西，就是在這裡做更改。
而在view最前面呼叫的class LaneCorrector(寫在lib/Panel/LaneCorrect.py)則是將網路輸出後經過後處理產生的點。修改這個文件增加其他功能來讓車道線的點點看起來更正確。