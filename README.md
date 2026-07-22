# AI 皮膚狀態偵測

以影像辨識初步判斷皮膚狀況的 Web 應用。上傳或拍攝皮膚照片後，系統會用白話說明告訴你「看起來像什麼」、「嚴不嚴重」以及「建議怎麼做」，降低醫學名詞帶來的閱讀門檻。

> **重要提醒**：本專案僅供學習與 AI 輔助初篩參考，**無法取代專業醫師診斷**。若有疑慮、症狀惡化或高風險結果，請盡快就醫。

## 功能特色

- 雙模型分析：皮膚癌相關病變 + 日常皮膚病
- 白話結果：顯示「一般的痣」「危險的皮膚癌」等易懂名稱
- 三項重點：這看起來像什麼、嚴重嗎、我該怎麼辦
- 視覺化風險：紅／黃／綠色塊搭配明確建議（盡快就醫／近期確認／持續觀察）
- 進階資訊：可展開查看各類別機率與專業名稱

## 技術架構

| 項目 | 說明 |
|------|------|
| 後端 | Python、Flask |
| AI | Hugging Face `transformers` pipeline |
| 影像 | Pillow |
| 前端 | HTML / CSS / JavaScript |

## 使用的 AI 模型

模型會在**第一次執行時自動下載**，快取於本機 Hugging Face 目錄（Windows 預設：`%USERPROFILE%\.cache\huggingface\hub`）。

| 用途 | Hugging Face 模型 |
|------|-------------------|
| 皮膚癌／痣等 | [Anwarkh1/Skin_Cancer-Image_Classification](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification) |
| 日常皮膚病 | [Jayanth2002/dinov2-base-finetuned-SkinDisease](https://huggingface.co/Jayanth2002/dinov2-base-finetuned-SkinDisease) |

## 專案結構

```
影像辨識/
├── app.py              # Flask 後端、模型載入、API、標籤對照
├── templates/
│   └── index.html      # 上傳介面與結果顯示
├── docs/
│   ├── 協作指南.md     # 分支開發、PR 送審流程（給協作者）
│   └── 管理員設定.md   # main 分支保護、審核 PR（給管理員）
├── .gitignore
├── LICENSE
└── README.md
```

## 團隊協作

本專案採 **分支 → Pull Request → 管理員審核** 後才合併 `main`，請勿直接 push 到 `main`。

| 角色 | 文件 |
|------|------|
| 協作者 | [docs/協作指南.md](docs/協作指南.md) |
| 管理員（Ohmomomo123） | [docs/管理員設定.md](docs/管理員設定.md) |

## 環境需求

- Python 3.10 以上（建議 3.12）
- 約 2～4 GB 磁碟空間（用於下載模型）
- 建議使用 CPU 即可（程式內 `device=-1`）

## 安裝步驟

```bash
# 1. 複製專案
git clone https://github.com/Ohmomomo123/skin.git
cd skin

# 2. 安裝依賴（建議使用虛擬環境）
pip install flask transformers torch pillow
```

## 啟動方式

```bash
python app.py
```

啟動後在瀏覽器開啟：

- 本機：<http://127.0.0.1:5000>
- 同一區域網路內其他裝置：<http://你的電腦IP:5000>

終端機出現 `模型載入完成！` 與 `Running on http://127.0.0.1:5000` 即表示服務已就緒。

> 首次啟動需下載兩個模型，可能需數分鐘，請耐心等候。

## API 說明

### `POST /predict`

上傳圖片進行分析。

- **參數**：`image`（multipart 檔案）
- **回傳範例**：

```json
{
  "simple_name": "一般的痣",
  "name": "黑色素細胞痣",
  "description": "皮膚上常見的痣，通常無害。",
  "severity_text": "✅ 持續觀察即可",
  "severity_level": "low",
  "action": "先觀察即可；若變大、變色、出血或形狀不規則，請到皮膚科檢查。",
  "confidence": 87.3,
  "all_predictions": [...]
}
```

## 常見問題

**Q：模型檔案在哪裡？為什麼 repo 裡沒有？**  
A：權重檔體積大，不納入 Git。執行 `app.py` 時會自動從 Hugging Face 下載到本機快取。

**Q：可以離線使用嗎？**  
A：模型下載完成後，在無需重新下載的情況下可離線推論；首次仍需網路。

**Q：準確度如何？**  
A：屬輔助初篩工具，受拍攝光線、角度、畫質影響。請以醫師診斷為準。

## 授權

本專案程式碼依 repository 內 `LICENSE` 授權條款使用。  
所使用的 Hugging Face 預訓練模型，請另參考各模型頁面上的授權說明。

## 免責聲明

本系統輸出結果不構成醫療建議、診斷或治療方案。開發者與貢獻者不對因使用本工具所產生的任何後果負責。如有健康疑慮，請諮詢合格醫療人員。
