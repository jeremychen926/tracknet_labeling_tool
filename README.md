# 影片標註工具：精簡指南

兩個 PyQt5 工具：
- `label_tool_multi.py` 多物體標註（輸出 JSON）
- `label_tool_v6.py` 單物體標註 + 事件（輸出 CSV）

選擇建議：多物體/需要 ID → 用 Multi；單一目標/需要事件 → 用 v6。

## 安裝
```bash
python -m pip install --upgrade pip
pip install PyQt5 opencv-python numpy pandas
```

## 如何執行
兩種方式擇一：

1) 互動式選片（程式會列出 `./video` 目錄中的檔案，已標註的會附註 (labeled)）：
```bash
python label_tool_multi.py
python label_tool_v6.py
```
輸入編號後開始標註，輸出資料自動放到 `json/` 或 `csv/`。

2) 直接指定影片路徑（相對或絕對）：
```bash
python label_tool_multi.py video/your_video.mp4
python label_tool_v6.py video/your_video.mp4
```
影片路徑的上層會被視為專案根，用於建立 `json/` 或 `csv/` 目錄。

## 功能重點
### Label Tool Multi（多物體）
- 多 ID 標註：左側清單顯示所有 ID，顏色循環使用 8 種預設色。
- 自動 ID 選擇：切幀時掃描上一幀最小未使用 ID；新增點後自動跳到下一個未使用 ID。
- 歷史疊加：顯示前 4 幀點位，透明度遞減，用於判斷移動方向；可勾選開關。
- 縮放播放：滑鼠滾輪或側邊滑桿縮放（1.0x ~ 約 5.6x），速度滑桿調整播放倍率（0.1x~2.0x）。
- 亮度微調：Brighten / Darken 按鈕在 0.1 ~ 3.0 間調整 alpha。
- 即時視圖：點位與 ID 號碼以小圓 + 文字顯示；前景點完全不透明。

### Label Tool v6（單物體）
- 單一物體：同一幀只有一組 (X,Y)；可見性自動建立，清除後 Visibility=0。
- 事件標記：數字 0~3 直接寫入當前幀的 Event 欄位（可自定語意）。
- 時間戳：依 FPS 自動換算 `Timestamp = frame_idx / fps`。
- 預測（右鍵）：收集最近 2~4 個可見點；≥3 用二次多項式擬合；2 點用線性外推；失敗時忽略。
- 像素微調：I/J/K/L 單像素修改座標（邊界保護避免超出畫面）。
- 歷史疊加：前 4 幀標註點淡化顯示紅點軌跡，輔助判斷連續性。
- 儲存行為：`S` 立即寫 CSV；退出（Q 或關閉視窗確認 Yes）自動保存。

## 快捷鍵（核心）
通用：`Z`/`X` 前後一幀，`D`/`F` ±50 幀跳轉，`Space` 播放/暫停，滾輪縮放。
Multi：左鍵新增，右鍵刪除最近點，`W`/`S` 切換 ID。
v6：左鍵標註，右鍵預測，`C` 清除，`0-3` 事件，`I/J/K/L` 微調，`S` 存，`Q` 存退出。

## 輸出
- Multi（JSON）：`json/<video_name>_track.json`
	- 結構：每幀 `Frame` + `Objects` 列表（`id, X, Y`）
	- 範例：
	```json
	[
	  {"Frame": 0, "Objects": [
	    {"id": 0, "X": 512, "Y": 384},
	    {"id": 1, "X": 640, "Y": 360}
	  ]},
	  {"Frame": 1, "Objects": [
	    {"id": 0, "X": 515, "Y": 382}
	  ]}
	]
	```
- v6（CSV）：`csv/<video_name>_ball.csv`
	- 欄位：`Frame, Visibility, X, Y, Z, Event, Timestamp`
	- 範例：
	```csv
	Frame,Visibility,X,Y,Z,Event,Timestamp
	0,0,0.0,0.0,0.0,0,0.0000
	1,1,512.0,384.0,0.0,0,0.0167
	2,1,515.0,382.0,0.0,1,0.0333
	```

## 小提示
- 兩個工具都會先把影片讀入記憶體；若出現 Killed，請降低解析度/時長或增加記憶體。
- Docker/WSL 執行 GUI 需設定 X11。僅標註不需 GPU，無 `/dev/dri` 也可執行。
