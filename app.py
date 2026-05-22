from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)

print("載入皮膚癌模型...")
cancer_model = pipeline(
    "image-classification",
    model="Anwarkh1/Skin_Cancer-Image_Classification",
    device=-1
)

print("載入日常皮膚病模型...")
daily_model = pipeline(
    "image-classification",
    model="Jayanth2002/dinov2-base-finetuned-SkinDisease",
    device=-1
)
print("模型載入完成！")

# ── 標籤對應表（專業名稱 + 白話名稱 + 說明）──
label_map = {
    "melanocytic_Nevi": {
        "name": "黑色素細胞痣", "simple_name": "一般的痣",
        "description": "皮膚上常見的痣，通常無害。",
        "risk": "低", "category": "病變",
        "action": "先觀察即可；若變大、變色、出血或形狀不規則，請到皮膚科檢查。"
    },
    "melanoma": {
        "name": "黑色素瘤", "simple_name": "危險的皮膚癌",
        "description": "惡性皮膚癌，需要立即就醫處理。",
        "risk": "高", "category": "病變",
        "action": "請盡快預約皮膚科或外科，不要拖延、不要自行處理。"
    },
    "basal_cell_carcinoma": {
        "name": "基底細胞癌", "simple_name": "一種皮膚癌",
        "description": "常見的皮膚癌，需要切除治療。",
        "risk": "高", "category": "病變",
        "action": "請盡快就醫，由醫師安排切除或進一步治療。"
    },
    "actinic_keratoses": {
        "name": "光化角化病", "simple_name": "日曬造成的皮膚變化",
        "description": "長期日曬導致，有癌變風險。",
        "risk": "中", "category": "病變",
        "action": "建議近期到皮膚科確認，並做好防曬。"
    },
    "benign_keratosis-like_lesions": {
        "name": "脂漏性角化病", "simple_name": "老人斑",
        "description": "年紀大常見的良性斑點，通常無害。",
        "risk": "低", "category": "病變",
        "action": "通常不用治療；若突然大量出現或快速變化，可請皮膚科評估。"
    },
    "dermatofibroma": {
        "name": "皮膚纖維瘤", "simple_name": "良性皮膚硬塊",
        "description": "常見的良性突起，通常無害。",
        "risk": "低", "category": "病變",
        "action": "可先觀察；若持續變大、疼痛，再請醫師檢查。"
    },
    "vascular_lesions": {
        "name": "血管病變", "simple_name": "血管相關的皮膚變化",
        "description": "通常為良性，但建議由醫師確認。",
        "risk": "中", "category": "病變",
        "action": "建議到皮膚科確認；若快速變大或出血，請盡快就醫。"
    },

    "Basal Cell Carcinoma": {
        "name": "基底細胞癌", "simple_name": "一種皮膚癌",
        "description": "常見的皮膚癌，需要切除治療。",
        "risk": "高", "category": "病變",
        "action": "請盡快就醫，由醫師安排切除或進一步治療。"
    },
    "Melanoma": {
        "name": "黑色素瘤", "simple_name": "危險的皮膚癌",
        "description": "惡性皮膚癌，需要立即就醫處理。",
        "risk": "高", "category": "病變",
        "action": "請盡快預約皮膚科或外科，不要拖延、不要自行處理。"
    },
    "Psoriasis": {
        "name": "乾癬", "simple_name": "慢性皮膚發炎",
        "description": "皮膚會脫屑，需長期治療。",
        "risk": "中", "category": "日常",
        "action": "建議到皮膚科就診，依醫囑用藥並避免過度抓搔。"
    },
    "Lichen Planus": {
        "name": "扁平苔癬", "simple_name": "皮膚發炎反應",
        "description": "會癢的皮膚發炎。",
        "risk": "中", "category": "日常",
        "action": "請皮膚科確認並治療，避免抓破造成感染。"
    },
    "Tinea Corporis": {
        "name": "錢癬", "simple_name": "黴菌感染",
        "description": "會傳染的黴菌，需要藥物治療。",
        "risk": "中", "category": "日常",
        "action": "請就醫使用抗黴菌藥物，避免共用毛巾、衣物。"
    },
    "Tinea Nigra": {
        "name": "黑色錢癬", "simple_name": "黴菌感染",
        "description": "黴菌引起的皮膚變色。",
        "risk": "中", "category": "日常",
        "action": "請皮膚科開立抗黴菌治療，保持患部乾燥。"
    },
    "Herpes Simplex": {
        "name": "單純皰疹", "simple_name": "病毒感染",
        "description": "嘴唇或皮膚的水泡，會復發。",
        "risk": "中", "category": "日常",
        "action": "避免摳抓或接觸患處；若疼痛明顯或反覆發作，請就醫。"
    },
    "Impetigo": {
        "name": "膿痂疹", "simple_name": "細菌感染",
        "description": "細菌引起，會傳染給其他人。",
        "risk": "中", "category": "日常",
        "action": "請就醫使用抗生素，注意手部清潔。"
    },
    "Pityriasis Rosea": {
        "name": "玫瑰糠疹", "simple_name": "皮膚紅疹",
        "description": "會自行痊癒，但需要追蹤。",
        "risk": "中", "category": "日常",
        "action": "多數會自行好轉；若持續很久或很癢，請皮膚科評估。"
    },
    "Molluscum Contagiosum": {
        "name": "傳染性軟疣", "simple_name": "病毒疣",
        "description": "病毒造成的小肉粒，會傳染。",
        "risk": "中", "category": "日常",
        "action": "避免抓破；若擴散或影響外觀，請皮膚科協助治療。"
    },
    "Pediculosis Capitis": {
        "name": "頭蝨", "simple_name": "寄生蟲",
        "description": "頭髮上的寄生蟲。",
        "risk": "低", "category": "日常",
        "action": "使用醫師建議的除蝨洗劑，並清洗帽子、枕頭、梳子。"
    },
    "Larva Migrans": {
        "name": "幼蟲遷徙症", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請就醫治療，並注意赤腳接觸泥土、沙灘的風險。"
    },
    "Tungiasis": {
        "name": "潛蚤病", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請醫師協助清除，注意足部衛生。"
    },
    "Leprosy Borderline": {
        "name": "邊緣型痲瘋", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科或感染科就醫，切勿自行判斷延誤。"
    },
    "Leprosy Lepromatous": {
        "name": "瘤型痲瘋", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科或感染科就醫，切勿自行判斷延誤。"
    },
    "Leprosy Tuberculoid": {
        "name": "結核樣型痲瘋", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科或感染科就醫，切勿自行判斷延誤。"
    },
    "Lupus Erythematosus Chronicus Discoides": {
        "name": "盤狀紅斑性狼瘡", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科就醫，並做好防曬。"
    },
    "Mycosis Fungoides": {
        "name": "蕈狀肉芽腫", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科就醫，不要自行用藥。"
    },
    "Neurofibromatosis": {
        "name": "神經纖維瘤", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請至皮膚科做完整評估與追蹤。"
    },
    "Darier_s Disease": {
        "name": "達里耶氏病", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請皮膚科長期追蹤，依醫囑保養。"
    },
    "Hailey-Hailey Disease": {
        "name": "海利氏病", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請皮膚科評估，避免摩擦、悶熱。"
    },
    "Epidermolysis Bullosa Pruriginosa": {
        "name": "癢性表皮鬆解症", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請至皮膚科或遺傳諮詢門診評估。"
    },
    "Porokeratosis Actinic": {
        "name": "光化性毛孔角化症", "simple_name": "日曬造成的皮膚變化",
        "description": "長期日曬導致，有癌變風險。",
        "risk": "中", "category": "病變",
        "action": "建議皮膚科確認，並做好防曬。"
    },
    "Papilomatosis Confluentes And Reticulate": {
        "name": "融合網狀乳頭瘤病", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "可到皮膚科確認；若影響外觀，可討論治療選項。"
    },
    "actinic keratosis": {
        "name": "光化角化病", "simple_name": "日曬造成的皮膚變化",
        "description": "長期日曬導致，有癌變風險。",
        "risk": "中", "category": "病變",
        "action": "建議近期到皮膚科確認，並做好防曬。"
    },
    "nevus": {
        "name": "黑色素細胞痣", "simple_name": "一般的痣",
        "description": "皮膚上常見的痣，通常無害。",
        "risk": "低", "category": "病變",
        "action": "先觀察即可；若變大、變色、出血或形狀不規則，請到皮膚科檢查。"
    },
    "pigmented benign keratosis": {
        "name": "色素性良性角化", "simple_name": "老人斑",
        "description": "年紀大常見的良性斑點，通常無害。",
        "risk": "低", "category": "病變",
        "action": "通常可先觀察；若快速變大或顏色改變，請皮膚科檢查。"
    },
    "seborrheic keratosis": {
        "name": "脂漏性角化", "simple_name": "老人斑",
        "description": "年紀大常見的良性斑點，通常無害。",
        "risk": "低", "category": "病變",
        "action": "通常不用治療；若突然大量出現，可請皮膚科評估。"
    },
    "squamous cell carcinoma": {
        "name": "鱗狀細胞癌", "simple_name": "一種皮膚癌",
        "description": "需盡快就醫處理。",
        "risk": "高", "category": "病變",
        "action": "請盡快就醫，由醫師安排切除或進一步治療。"
    },
    "vascular lesion": {
        "name": "血管病變", "simple_name": "血管相關的皮膚變化",
        "description": "通常為良性，但建議由醫師確認。",
        "risk": "中", "category": "病變",
        "action": "建議到皮膚科確認；若快速變大或出血，請盡快就醫。"
    },
}

DEFAULT_LABEL = {
    "name": "未能明確辨識",
    "simple_name": "罕見皮膚問題",
    "description": "建議就醫由醫師判斷。",
    "risk": "中",
    "category": "未分類",
    "action": "請到皮膚科門診，由醫師搭配實際檢查確認。"
}

# 嚴重程度（視覺化用語）
severity_display = {
    "高": {"text": "⚠️ 建議盡快就醫", "level": "high"},
    "中": {"text": "💡 建議近期就醫確認", "level": "medium"},
    "低": {"text": "✅ 持續觀察即可", "level": "low"},
}


def _label_info(label_key):
    if label_key in label_map:
        return label_map[label_key]
    return {**DEFAULT_LABEL, "name": label_key}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')

    cancer_results = cancer_model(image)
    daily_results = daily_model(image)

    top_cancer = cancer_results[0]
    top_daily = daily_results[0]

    if top_cancer['score'] >= top_daily['score']:
        winner = top_cancer
        source = "cancer_model"
    else:
        winner = top_daily
        source = "daily_model"

    info = _label_info(winner['label'])
    severity = severity_display.get(info['risk'], severity_display['中'])
    confidence = round(winner['score'] * 100, 1)

    all_predictions = []
    for r in cancer_results[:3]:
        i = _label_info(r['label'])
        all_predictions.append({
            "simple_name": i.get('simple_name', '罕見皮膚問題'),
            "name": i['name'],
            "description": i.get('description', ''),
            "prob": round(r['score'] * 100, 1)
        })
    for r in daily_results[:3]:
        i = _label_info(r['label'])
        all_predictions.append({
            "simple_name": i.get('simple_name', '罕見皮膚問題'),
            "name": i['name'],
            "description": i.get('description', ''),
            "prob": round(r['score'] * 100, 1)
        })

    all_predictions.sort(key=lambda x: x['prob'], reverse=True)

    return jsonify({
        "name": info['name'],
        "simple_name": info.get('simple_name', DEFAULT_LABEL['simple_name']),
        "description": info.get('description', DEFAULT_LABEL['description']),
        "severity_text": severity['text'],
        "severity_level": severity['level'],
        "action": info.get('action', DEFAULT_LABEL['action']),
        "confidence": confidence,
        "source": source,
        "all_predictions": all_predictions[:6],
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
