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
        "action": "先觀察即可；若變大、變色、出血或形狀不規則，請到皮膚科檢查。",
        "care": "每天塗抹 SPF 30+ 防曬乳，避免紫外線刺激；定期自我觀察痣的形狀與顏色變化。",
        "avoid": "避免過度摩擦或抓傷痣；避免長時間直接曝曬強烈陽光。",
        "see_doctor": "痣的邊緣不規則、顏色不均、直徑超過 6 mm，或出血、快速變大時，請就醫。"
    },
    "melanoma": {
        "name": "黑色素瘤", "simple_name": "危險的皮膚癌",
        "description": "惡性皮膚癌，需要立即就醫處理。",
        "risk": "高", "category": "病變",
        "action": "請盡快預約皮膚科或外科，不要拖延、不要自行處理。",
        "care": "立即就醫並遵醫囑治療；術後定期回診追蹤，防止復發。",
        "avoid": "不要自行處理、遮掩或使用偏方；避免日曬延誤病情。",
        "see_doctor": "懷疑為黑色素瘤時請立即就醫，不要等待觀察。"
    },
    "basal_cell_carcinoma": {
        "name": "基底細胞癌", "simple_name": "一種皮膚癌",
        "description": "常見的皮膚癌，需要切除治療。",
        "risk": "高", "category": "病變",
        "action": "請盡快就醫，由醫師安排切除或進一步治療。",
        "care": "術後保持傷口清潔，遵醫囑換藥；定期回診追蹤是否復發。",
        "avoid": "避免繼續日曬患部；不要自行剪除、灼燒或塗抹偏方。",
        "see_doctor": "發現疑似病灶時請立即就醫，不要自行觀察等待。"
    },
    "actinic_keratoses": {
        "name": "光化角化病", "simple_name": "日曬造成的皮膚變化",
        "description": "長期日曬導致，有癌變風險。",
        "risk": "中", "category": "病變",
        "action": "建議近期到皮膚科確認，並做好防曬。",
        "care": "每天塗抹 SPF 50+ 防曬乳；外出穿長袖、戴帽子遮蔽患部。",
        "avoid": "避免正午陽光直曬；不要自行剝除或刮去角化皮屑。",
        "see_doctor": "病灶增多、變厚變硬、出血，或形成潰瘍時，請盡快就醫。"
    },
    "benign_keratosis-like_lesions": {
        "name": "脂漏性角化病", "simple_name": "老人斑",
        "description": "年紀大常見的良性斑點，通常無害。",
        "risk": "低", "category": "病變",
        "action": "通常不用治療；若突然大量出現或快速變化，可請皮膚科評估。",
        "care": "保持皮膚滋潤，減少衣物摩擦；日常做好防曬。",
        "avoid": "避免硬抓或用指甲刮除，容易造成感染或出血。",
        "see_doctor": "短期內突然大量增生、顏色急劇改變，或出血時，請就醫確認。"
    },
    "dermatofibroma": {
        "name": "皮膚纖維瘤", "simple_name": "良性皮膚硬塊",
        "description": "常見的良性突起，通常無害。",
        "risk": "低", "category": "病變",
        "action": "可先觀察；若持續變大、疼痛，再請醫師檢查。",
        "care": "避免衣物或配件反覆摩擦患部；日常保濕維持皮膚屏障。",
        "avoid": "不要嘗試自行擠壓、針刺或割除，容易感染或留疤。",
        "see_doctor": "腫塊持續變大、出現明顯疼痛，或形狀改變時，請就醫評估。"
    },
    "vascular_lesions": {
        "name": "血管病變", "simple_name": "血管相關的皮膚變化",
        "description": "通常為良性，但建議由醫師確認。",
        "risk": "中", "category": "病變",
        "action": "建議到皮膚科確認；若快速變大或出血，請盡快就醫。",
        "care": "保護患部避免外傷；若有搔癢可冷敷緩解不適。",
        "avoid": "避免摩擦或按壓患部；不要自行刺破或灼燒。",
        "see_doctor": "快速擴大、出血不止，或顏色突然改變時，請盡快就醫。"
    },

    "Basal Cell Carcinoma": {
        "name": "基底細胞癌", "simple_name": "一種皮膚癌",
        "description": "常見的皮膚癌，需要切除治療。",
        "risk": "高", "category": "病變",
        "action": "請盡快就醫，由醫師安排切除或進一步治療。",
        "care": "術後保持傷口清潔，遵醫囑換藥；定期回診追蹤是否復發。",
        "avoid": "避免繼續日曬患部；不要自行剪除、灼燒或塗抹偏方。",
        "see_doctor": "發現疑似病灶時請立即就醫，不要自行觀察等待。"
    },
    "Melanoma": {
        "name": "黑色素瘤", "simple_name": "危險的皮膚癌",
        "description": "惡性皮膚癌，需要立即就醫處理。",
        "risk": "高", "category": "病變",
        "action": "請盡快預約皮膚科或外科，不要拖延、不要自行處理。",
        "care": "立即就醫並遵醫囑治療；術後定期回診追蹤，防止復發。",
        "avoid": "不要自行處理、遮掩或使用偏方；避免日曬延誤病情。",
        "see_doctor": "懷疑為黑色素瘤時請立即就醫，不要等待觀察。"
    },
    "Psoriasis": {
        "name": "乾癬", "simple_name": "慢性皮膚發炎",
        "description": "皮膚會脫屑，需長期治療。",
        "risk": "中", "category": "日常",
        "action": "建議到皮膚科就診，依醫囑用藥並避免過度抓搔。",
        "care": "每天使用無香精保濕乳液滋潤皮膚；洗澡用溫水及溫和沐浴乳，時間不宜過長。",
        "avoid": "避免過度搔抓、壓力過大及皮膚外傷（可能誘發同形反應）；避免酒精與辛辣食物。",
        "see_doctor": "病灶大範圍擴散、出現關節腫痛，或用藥後數週無改善時，請回診調整治療。"
    },
    "Lichen Planus": {
        "name": "扁平苔癬", "simple_name": "皮膚發炎反應",
        "description": "會癢的皮膚發炎。",
        "risk": "中", "category": "日常",
        "action": "請皮膚科確認並治療，避免抓破造成感染。",
        "care": "保持皮膚滋潤；穿著寬鬆衣物減少對患部的摩擦刺激。",
        "avoid": "避免抓破皮膚形成潰瘍；避免使用強效清潔產品過度清洗患部。",
        "see_doctor": "口腔黏膜或指甲也受影響，或搔癢嚴重影響睡眠時，請告知醫師。"
    },
    "Tinea Corporis": {
        "name": "錢癬", "simple_name": "黴菌感染",
        "description": "會傳染的黴菌，需要藥物治療。",
        "risk": "中", "category": "日常",
        "action": "請就醫使用抗黴菌藥物，避免共用毛巾、衣物。",
        "care": "保持患部乾燥通風；完成醫師開立的完整抗黴菌療程（通常 2–4 週）。",
        "avoid": "不要共用毛巾、衣物；避免悶熱潮濕環境，如不透氣的緊身衣。",
        "see_doctor": "用藥兩週後未改善，或病灶持續擴大、化膿時，請回診調整用藥。"
    },
    "Tinea Nigra": {
        "name": "黑色錢癬", "simple_name": "黴菌感染",
        "description": "黴菌引起的皮膚變色。",
        "risk": "中", "category": "日常",
        "action": "請皮膚科開立抗黴菌治療，保持患部乾燥。",
        "care": "按醫囑塗抹抗黴菌藥物；保持手腳乾燥，出汗後及時擦乾。",
        "avoid": "避免潮濕環境；不要自行用去角質產品刮除，以免刺激皮膚。",
        "see_doctor": "治療後顏色未消退，或病灶持續擴大時，請回診確認。"
    },
    "Herpes Simplex": {
        "name": "單純皰疹", "simple_name": "病毒感染",
        "description": "嘴唇或皮膚的水泡，會復發。",
        "risk": "中", "category": "日常",
        "action": "避免摳抓或接觸患處；若疼痛明顯或反覆發作，請就醫。",
        "care": "水泡期間保持患部清潔乾燥；觸碰後立即洗手，避免傳染眼睛或他人。",
        "avoid": "發作期間避免親密接觸（如接吻）傳染他人；不要摳破水泡，防止繼發感染。",
        "see_doctor": "水泡擴散至眼周、反覆頻繁發作（每月超過一次），或伴隨高燒時，請就醫。"
    },
    "Impetigo": {
        "name": "膿痂疹", "simple_name": "細菌感染",
        "description": "細菌引起，會傳染給其他人。",
        "risk": "中", "category": "日常",
        "action": "請就醫使用抗生素，注意手部清潔。",
        "care": "輕輕用生理食鹽水清潔患部後覆蓋紗布；勤洗手防止自我傳染與傳染他人。",
        "avoid": "不要抓破結痂；不要與他人共用毛巾、餐具、衣物。",
        "see_doctor": "服藥 48 小時後仍持續擴散、出現發燒，或淋巴結腫大時，請立即回診。"
    },
    "Pityriasis Rosea": {
        "name": "玫瑰糠疹", "simple_name": "皮膚紅疹",
        "description": "會自行痊癒，但需要追蹤。",
        "risk": "中", "category": "日常",
        "action": "多數會自行好轉；若持續很久或很癢，請皮膚科評估。",
        "care": "穿寬鬆透氣衣物；可使用保濕乳液緩解搔癢，洗澡水溫不宜過熱。",
        "avoid": "避免過熱洗澡或劇烈運動（體溫上升會加重搔癢）；避免使用刺激性肥皂。",
        "see_doctor": "超過 3 個月仍未消退，或搔癢難以忍受、影響日常生活時，請就醫。"
    },
    "Molluscum Contagiosum": {
        "name": "傳染性軟疣", "simple_name": "病毒疣",
        "description": "病毒造成的小肉粒，會傳染。",
        "risk": "中", "category": "日常",
        "action": "避免抓破；若擴散或影響外觀，請皮膚科協助治療。",
        "care": "保持患部清潔；觸碰後立即洗手，避免自我接種到其他部位。",
        "avoid": "不要抓破或擠壓疣體；避免共用浴巾；游泳時注意防護以免傳染他人。",
        "see_doctor": "廣泛擴散（超過 20 顆）、位於眼周，或免疫力低下者，請儘早就醫治療。"
    },
    "Pediculosis Capitis": {
        "name": "頭蝨", "simple_name": "寄生蟲",
        "description": "頭髮上的寄生蟲。",
        "risk": "低", "category": "日常",
        "action": "使用醫師建議的除蝨洗劑，並清洗帽子、枕頭、梳子。",
        "care": "使用除蝨洗劑後，7–10 天再重複一次；清洗所有帽子、枕套、梳子（60°C 以上）。",
        "avoid": "避免頭部直接接觸他人（如共用枕頭）；不要共用梳子、髮夾、頭巾。",
        "see_doctor": "完整療程後仍可在頭皮發現活蝨，或頭皮出現紅腫化膿的繼發感染時。"
    },
    "Larva Migrans": {
        "name": "幼蟲遷徙症", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請就醫治療，並注意赤腳接觸泥土、沙灘的風險。",
        "care": "遵醫囑服用抗寄生蟲藥物；保持患部清潔，避免繼發感染。",
        "avoid": "在熱帶地區或海灘避免赤腳踩踏沙地或泥土；不要自行用針挑除。",
        "see_doctor": "皮膚出現移動性紅色線狀痕跡、伴隨發燒或腹痛時，請立即就醫。"
    },
    "Tungiasis": {
        "name": "潛蚤病", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請醫師協助清除，注意足部衛生。",
        "care": "醫師清除後保持傷口清潔並定期換藥；穿著密閉鞋子預防再感染。",
        "avoid": "在蚤病流行地區不要赤腳行走；不要自行用尖物挑除，容易感染甚至破傷風。",
        "see_doctor": "傷口出現紅腫化膿、多處同時感染，或伴隨發燒時，請立即就醫。"
    },
    "Leprosy Borderline": {
        "name": "邊緣型痲瘋", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科或感染科就醫，切勿自行判斷延誤。",
        "care": "遵從長期聯合用藥（MDT）療程；保護感覺麻木的手腳部位，避免燙傷或割傷。",
        "avoid": "不要中斷藥物治療；避免傷口感染導致永久殘疾。",
        "see_doctor": "出現新的皮膚斑塊、神經疼痛，或手腳感覺異常時，請立即回診。"
    },
    "Leprosy Lepromatous": {
        "name": "瘤型痲瘋", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科或感染科就醫，切勿自行判斷延誤。",
        "care": "遵從長期聯合用藥（MDT）療程；保護感覺麻木的手腳部位，避免燙傷或割傷。",
        "avoid": "不要中斷藥物治療；避免傷口感染導致永久殘疾。",
        "see_doctor": "出現新的皮膚斑塊、神經疼痛，或手腳感覺異常時，請立即回診。"
    },
    "Leprosy Tuberculoid": {
        "name": "結核樣型痲瘋", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科或感染科就醫，切勿自行判斷延誤。",
        "care": "遵從長期聯合用藥（MDT）療程；保護感覺麻木的手腳部位，避免燙傷或割傷。",
        "avoid": "不要中斷藥物治療；避免傷口感染導致永久殘疾。",
        "see_doctor": "出現新的皮膚斑塊、神經疼痛，或手腳感覺異常時，請立即回診。"
    },
    "Lupus Erythematosus Chronicus Discoides": {
        "name": "盤狀紅斑性狼瘡", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科就醫，並做好防曬。",
        "care": "嚴格防曬（SPF 50+，物理性防曬為主）；按醫囑塗抹藥膏並定期回診追蹤。",
        "avoid": "避免任何形式的日曬（包含室內紫外線燈）；避免壓力與感染誘發惡化。",
        "see_doctor": "病灶快速擴大，或出現全身症狀（關節痛、發燒、疲倦）時，請立即就醫。"
    },
    "Mycosis Fungoides": {
        "name": "蕈狀肉芽腫", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請盡快至皮膚科就醫，不要自行用藥。",
        "care": "遵醫囑接受治療；保持皮膚滋潤，減少搔癢不適。",
        "avoid": "不要自行使用類固醇藥膏遮掩症狀；避免延誤就醫，此病需及早確診分期。",
        "see_doctor": "皮疹快速擴展、出現淋巴腫大、體重減輕，或持續發燒時，請立即就醫。"
    },
    "Neurofibromatosis": {
        "name": "神經纖維瘤", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "高", "category": "病變",
        "action": "請至皮膚科做完整評估與追蹤。",
        "care": "定期接受皮膚科與神經科追蹤；避免腫瘤部位受壓或外傷。",
        "avoid": "不要自行切除腫瘤；避免過度日曬刺激皮膚。",
        "see_doctor": "腫瘤快速變大、出現明顯疼痛，或伴隨視力、聽力、學習異常時，請就醫。"
    },
    "Darier_s Disease": {
        "name": "達里耶氏病", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請皮膚科長期追蹤，依醫囑保養。",
        "care": "保持皮膚清潔乾燥；使用無香精的溫和保濕乳液，減少摩擦刺激。",
        "avoid": "避免高溫悶熱環境（會加重症狀）；避免外傷摩擦及強烈日曬。",
        "see_doctor": "皮膚出現化膿、紅熱感染跡象，或症狀突然明顯惡化時，請就醫。"
    },
    "Hailey-Hailey Disease": {
        "name": "海利氏病", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請皮膚科評估，避免摩擦、悶熱。",
        "care": "保持患部乾燥通風；穿著寬鬆透氣衣物，避免皮膚皺褶處悶濕。",
        "avoid": "避免摩擦、流汗、悶熱（主要誘發因素）；不要穿緊身衣物壓迫患部。",
        "see_doctor": "患部出現明顯化膿、紅熱，或病灶快速擴展時，請就醫處理感染。"
    },
    "Epidermolysis Bullosa Pruriginosa": {
        "name": "癢性表皮鬆解症", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "請至皮膚科或遺傳諮詢門診評估。",
        "care": "使用非黏性敷料保護皮膚；保持室溫涼爽，減少水泡形成。",
        "avoid": "避免摩擦或碰撞皮膚；不要強行撕除黏附敷料，以免撕裂皮膚。",
        "see_doctor": "傷口出現化膿異味，或突然大面積水泡形成時，請立即就醫。"
    },
    "Porokeratosis Actinic": {
        "name": "光化性毛孔角化症", "simple_name": "日曬造成的皮膚變化",
        "description": "長期日曬導致，有癌變風險。",
        "risk": "中", "category": "病變",
        "action": "建議皮膚科確認，並做好防曬。",
        "care": "每天塗抹 SPF 50+ 防曬乳；外出穿長袖、戴帽子保護患部。",
        "avoid": "避免正午陽光直曬；不要自行剝除角化環，以免刺激或惡化。",
        "see_doctor": "病灶增多變大、出現潰瘍或出血時，請盡快就醫排除癌變。"
    },
    "Papilomatosis Confluentes And Reticulate": {
        "name": "融合網狀乳頭瘤病", "simple_name": "罕見皮膚問題",
        "description": "建議就醫由醫師判斷。",
        "risk": "中", "category": "日常",
        "action": "可到皮膚科確認；若影響外觀，可討論治療選項。",
        "care": "保持患部清潔；若有異味，可使用抗菌清潔用品輕柔清洗。",
        "avoid": "避免過度日曬；不要用力摩擦患部，以免刺激擴散。",
        "see_doctor": "病灶擴散至頸部、腋下等大範圍，或抗生素治療後無反應時，請回診。"
    },
    "actinic keratosis": {
        "name": "光化角化病", "simple_name": "日曬造成的皮膚變化",
        "description": "長期日曬導致，有癌變風險。",
        "risk": "中", "category": "病變",
        "action": "建議近期到皮膚科確認，並做好防曬。",
        "care": "每天塗抹 SPF 50+ 防曬乳；外出穿長袖、戴帽子遮蔽患部。",
        "avoid": "避免正午陽光直曬；不要自行剝除或刮去角化皮屑。",
        "see_doctor": "病灶增多、變厚變硬、出血，或形成潰瘍時，請盡快就醫。"
    },
    "nevus": {
        "name": "黑色素細胞痣", "simple_name": "一般的痣",
        "description": "皮膚上常見的痣，通常無害。",
        "risk": "低", "category": "病變",
        "action": "先觀察即可；若變大、變色、出血或形狀不規則，請到皮膚科檢查。",
        "care": "每天塗抹 SPF 30+ 防曬乳；定期自我觀察痣的形狀、大小與顏色變化。",
        "avoid": "避免過度摩擦或抓傷痣；避免長時間直接曝曬強烈陽光。",
        "see_doctor": "痣的邊緣不規則、顏色不均、直徑超過 6 mm，或出血、快速變大時，請就醫。"
    },
    "pigmented benign keratosis": {
        "name": "色素性良性角化", "simple_name": "老人斑",
        "description": "年紀大常見的良性斑點，通常無害。",
        "risk": "低", "category": "病變",
        "action": "通常可先觀察；若快速變大或顏色改變，請皮膚科檢查。",
        "care": "保持皮膚滋潤，減少衣物摩擦；日常做好防曬。",
        "avoid": "避免硬抓或用指甲刮除，容易造成感染或出血。",
        "see_doctor": "短期內顏色急劇改變、出血，或突然快速增大時，請就醫確認。"
    },
    "seborrheic keratosis": {
        "name": "脂漏性角化", "simple_name": "老人斑",
        "description": "年紀大常見的良性斑點，通常無害。",
        "risk": "低", "category": "病變",
        "action": "通常不用治療；若突然大量出現，可請皮膚科評估。",
        "care": "保持皮膚滋潤，減少衣物摩擦；日常做好防曬。",
        "avoid": "避免硬抓或用指甲刮除，容易造成感染或出血。",
        "see_doctor": "短期內突然大量增生、顏色急劇改變，或出血時，請就醫確認。"
    },
    "squamous cell carcinoma": {
        "name": "鱗狀細胞癌", "simple_name": "一種皮膚癌",
        "description": "需盡快就醫處理。",
        "risk": "高", "category": "病變",
        "action": "請盡快就醫，由醫師安排切除或進一步治療。",
        "care": "立即就醫並遵醫囑治療；術後定期回診，做好防曬防止復發。",
        "avoid": "避免繼續日曬患部；不要自行使用偏方處理或延誤就醫。",
        "see_doctor": "發現疑似病灶時請立即就醫，不要自行觀察等待。"
    },
    "vascular lesion": {
        "name": "血管病變", "simple_name": "血管相關的皮膚變化",
        "description": "通常為良性，但建議由醫師確認。",
        "risk": "中", "category": "病變",
        "action": "建議到皮膚科確認；若快速變大或出血，請盡快就醫。",
        "care": "保護患部避免外傷；若有搔癢可冷敷緩解不適。",
        "avoid": "避免摩擦或按壓患部；不要自行刺破或灼燒。",
        "see_doctor": "快速擴大、出血不止，或顏色突然改變時，請盡快就醫。"
    },
}

DEFAULT_LABEL = {
    "name": "未能明確辨識",
    "simple_name": "罕見皮膚問題",
    "description": "建議就醫由醫師判斷。",
    "risk": "中",
    "category": "未分類",
    "action": "請到皮膚科門診，由醫師搭配實際檢查確認。",
    "care": "保持患部清潔，避免自行處理；日常做好防曬與皮膚保濕。",
    "avoid": "避免抓破或自行用藥；不要忽視持續變化的皮膚症狀。",
    "see_doctor": "症狀持續超過兩週、快速惡化，或伴隨疼痛、出血時，請就醫。"
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
        "care": info.get('care', DEFAULT_LABEL['care']),
        "avoid": info.get('avoid', DEFAULT_LABEL['avoid']),
        "see_doctor": info.get('see_doctor', DEFAULT_LABEL['see_doctor']),
        "confidence": confidence,
        "source": source,
        "all_predictions": all_predictions[:6],
    })


@app.route('/supported')
def supported():
    seen = set()
    risk_to_level = {"高": "high", "中": "medium", "低": "low"}
    groups = {"high": [], "medium": [], "low": []}

    for info in label_map.values():
        name = info["name"]
        if name in seen:
            continue
        seen.add(name)
        level = risk_to_level.get(info.get("risk", "中"), "medium")
        groups[level].append({
            "simple_name": info.get("simple_name", ""),
            "name": name,
        })

    for level in groups:
        groups[level].sort(key=lambda x: x["simple_name"])

    return jsonify(groups)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
