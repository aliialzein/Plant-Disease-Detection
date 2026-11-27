import json
from pathlib import Path

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease AI", page_icon="🌱", layout="centered")

st.title("🌱 Plant Disease AI")
st.caption("Nhận diện 15 nhóm tình trạng bệnh trên cà chua, khoai tây và ớt chuông.")

IMG_SIZE = 128
MODEL_DIR = Path("model")
NEW_MODEL_PATH = MODEL_DIR / "best_model.keras"
LEGACY_MODEL_PATH = MODEL_DIR / "best_model.h5"
MODEL_PATH = str(NEW_MODEL_PATH if NEW_MODEL_PATH.exists() else LEGACY_MODEL_PATH)

# IMPORTANT: this order matches Keras flow_from_directory alphabetical class_indices
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy",
]

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "name": "🌶️ Ớt chuông — Bệnh đốm vi khuẩn",
        "symptoms": "Đốm nhỏ màu nâu/đen trên lá, đôi khi có quầng vàng.",
        "treatment": "Loại bỏ lá bị bệnh, giữ lá khô, tưới vào gốc và tăng thông thoáng. Một số thuốc gốc đồng có thể được dùng để quản lý bệnh nếu nhãn sản phẩm cho phép dùng trên ớt.",
        "active": "Hoạt chất thường gặp: hợp chất đồng (copper). Vi khuẩn có thể kháng đồng, nên không xem đây là thuốc chữa chắc chắn.",
    },
    "Pepper__bell___healthy": {
        "name": "🌶️ Ớt chuông — Khỏe mạnh",
        "symptoms": "Lá xanh, phát triển bình thường, không thấy triệu chứng bệnh rõ ràng.",
        "treatment": "Chưa thấy dấu hiệu bệnh thuộc các class của model. Tiếp tục theo dõi cây.",
        "active": "Không cần thuốc theo kết quả này.",
    },
    "Potato___Early_blight": {
        "name": "🥔 Khoai tây — Bệnh cháy lá sớm",
        "symptoms": "Đốm nâu tròn, thường có vòng đồng tâm; lá có thể vàng và khô dần.",
        "treatment": "Loại bỏ lá bệnh, giữ tán lá khô, tăng thông thoáng và vệ sinh tàn dư cây. Nếu cần thuốc, dùng thuốc trừ nấm được đăng ký cho khoai tây và đúng bệnh trên nhãn.",
        "active": "Hoạt chất thường được dùng trong chương trình quản lý bệnh có thể gồm chlorothalonil hoặc nhóm thuốc trừ nấm khác tùy nhãn và khu vực.",
    },
    "Potato___Late_blight": {
        "name": "🥔 Khoai tây — Bệnh mốc sương",
        "symptoms": "Vết nâu sẫm/đen lớn, lan nhanh; khi ẩm có thể xuất hiện lớp mốc trắng.",
        "treatment": "Cách ly và loại bỏ phần bệnh, giữ tán lá khô và theo dõi sát. Mốc sương phát triển rất nhanh; nếu dùng thuốc cần thuốc được đăng ký cho bệnh này và phải phun theo nhãn.",
        "active": "Thuốc chuyên quản lý Phytophthora thường được dùng theo chương trình luân phiên nhóm hoạt chất; không nên tự pha hoặc tăng liều.",
    },
    "Potato___healthy": {
        "name": "🥔 Khoai tây — Khỏe mạnh",
        "symptoms": "Lá xanh bình thường, không thấy dấu hiệu bệnh rõ ràng.",
        "treatment": "Chưa thấy dấu hiệu bệnh thuộc các class của model. Tiếp tục theo dõi cây.",
        "active": "Không cần thuốc theo kết quả này.",
    },
    "Tomato__Bacterial_spot": {
        "name": "🍅 Cà chua — Bệnh đốm vi khuẩn",
        "symptoms": "Đốm nhỏ sẫm màu, thường có quầng vàng; vết bệnh nặng có thể làm lá rụng.",
        "treatment": "Loại bỏ lá bệnh, tránh tưới phun lên lá, giữ khoảng cách thông thoáng và vệ sinh dụng cụ. Một số thuốc gốc đồng có thể được dùng để quản lý bệnh nếu nhãn cho phép.",
        "active": "Hoạt chất thường gặp: hợp chất đồng (copper). Kháng đồng đã được ghi nhận nên hiệu quả có thể hạn chế.",
    },
    "Tomato__Early_blight": {
        "name": "🍅 Cà chua — Bệnh cháy lá sớm",
        "symptoms": "Đốm nâu với các vòng đồng tâm, thường xuất hiện trước ở lá già.",
        "treatment": "Tỉa lá bệnh ở mức vừa phải, giữ lá khô, tưới vào gốc, tăng thông thoáng và vệ sinh tàn dư. Nếu cần thuốc, chọn thuốc trừ nấm được đăng ký cho cà chua và bệnh này.",
        "active": "Chlorothalonil là một hoạt chất thường xuất hiện trong chương trình quản lý cháy lá sớm; luôn kiểm tra nhãn sản phẩm tại Việt Nam.",
    },
    "Tomato__Late_blight": {
        "name": "🍅 Cà chua — Bệnh mốc sương",
        "symptoms": "Vết nâu lớn lan nhanh; trong điều kiện ẩm có thể xuất hiện lớp mốc trắng.",
        "treatment": "Loại bỏ phần bệnh, giảm ẩm trên tán lá, tăng thông thoáng và theo dõi sát. Nếu bệnh lan nhanh, dùng thuốc được đăng ký cho late blight/Phytophthora trên cà chua theo đúng nhãn.",
        "active": "Cần thuốc chuyên quản lý Phytophthora; hoạt chất và lịch phun phải theo nhãn sản phẩm và tình hình kháng thuốc.",
    },
    "Tomato__Leaf_Mold": {
        "name": "🍅 Cà chua — Bệnh mốc lá",
        "symptoms": "Mặt trên lá có vùng vàng; mặt dưới có thể có lớp mốc xanh vàng đến nâu ôliu.",
        "treatment": "Giảm độ ẩm, tăng thông gió, tránh để lá ướt kéo dài và loại bỏ tàn dư bệnh. Một số sản phẩm gốc đồng có thể được dùng nếu được đăng ký cho bệnh này.",
        "active": "Copper hydroxide có dữ liệu sử dụng trong quản lý mốc lá; phải kiểm tra nhãn sản phẩm.",
    },
    "Tomato__Septoria_leaf_spot": {
        "name": "🍅 Cà chua — Bệnh đốm lá Septoria",
        "symptoms": "Nhiều đốm nhỏ tròn, tâm xám/nâu nhạt, viền sẫm; có thể có chấm đen nhỏ ở giữa.",
        "treatment": "Tỉa lá bệnh, giữ tán lá khô, tưới vào gốc và vệ sinh tàn dư. Nếu cần thuốc, dùng thuốc trừ nấm được đăng ký cho Septoria trên cà chua.",
        "active": "Chlorothalonil là một hoạt chất thường được dùng trong chương trình quản lý bệnh đốm lá; kiểm tra nhãn và thời gian cách ly.",
    },
    "Tomato__Spider_mites_Two_spotted_spider_mite": {
        "name": "🍅 Cà chua — Nhện đỏ hai chấm",
        "symptoms": "Lá có nhiều chấm vàng li ti, bạc màu; nặng có thể xuất hiện tơ nhện mảnh.",
        "treatment": "Kiểm tra mặt dưới lá, rửa/tưới nhẹ để giảm mật số và bảo vệ thiên địch. Xà phòng diệt côn trùng hoặc dầu làm vườn có thể hiệu quả khi tiếp xúc trực tiếp với nhện.",
        "active": "Xà phòng diệt côn trùng (potassium salts of fatty acids) hoặc dầu làm vườn/neem; cần phun trúng mặt dưới lá và lặp lại theo nhãn.",
    },
    "Tomato__Target_Spot": {
        "name": "🍅 Cà chua — Bệnh đốm vòng (Target Spot)",
        "symptoms": "Đốm nâu có các vòng đồng tâm giống hình bia; bệnh nặng có thể làm lá vàng và rụng.",
        "treatment": "Giữ tán lá khô, tăng thông thoáng, vệ sinh lá bệnh và tàn dư. Nếu cần thuốc, dùng thuốc trừ nấm được đăng ký cho target spot trên cà chua.",
        "active": "Hoạt chất phụ thuộc sản phẩm và khu vực; nên chọn theo nhãn đăng ký thay vì tự dùng một thuốc cố định.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "🍅 Cà chua — Virus xoăn vàng lá",
        "symptoms": "Lá vàng, nhỏ lại và cuốn/xoăn; cây có thể sinh trưởng kém.",
        "treatment": "Không có thuốc diệt virus để chữa cây đã nhiễm. Loại bỏ cây bệnh nặng, kiểm soát côn trùng môi giới và dùng cây giống sạch bệnh.",
        "active": "Không có thuốc chữa virus trực tiếp; quản lý vector và vệ sinh cây trồng là chính.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "🍅 Cà chua — Virus khảm",
        "symptoms": "Lá có các mảng xanh đậm/xanh nhạt xen kẽ dạng khảm, đôi khi bị biến dạng.",
        "treatment": "Không có thuốc chữa virus trực tiếp. Loại bỏ cây nhiễm nặng, vệ sinh tay/dụng cụ và dùng hạt/cây giống sạch bệnh.",
        "active": "Không có thuốc diệt virus trực tiếp.",
    },
    "Tomato__healthy": {
        "name": "🍅 Cà chua — Khỏe mạnh",
        "symptoms": "Lá xanh tương đối đồng đều, không có triệu chứng bệnh rõ ràng.",
        "treatment": "Chưa thấy dấu hiệu bệnh thuộc các class của model. Tiếp tục theo dõi cây.",
        "active": "Không cần thuốc theo kết quả này.",
    },
}

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Use the exact output order saved alongside a newly trained model.
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
if CLASS_NAMES_PATH.exists():
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        CLASS_NAMES = json.load(file)

missing_info = [name for name in CLASS_NAMES if name not in DISEASE_INFO]
if missing_info:
    st.error(f"Thiếu thông tin hiển thị cho các lớp: {missing_info}")
    st.stop()

model = load_model(MODEL_PATH)
output_classes = int(model.output_shape[-1])
if output_classes != len(CLASS_NAMES):
    st.error(
        f"Model có {output_classes} đầu ra nhưng danh sách lớp có {len(CLASS_NAMES)} mục. "
        "Vui lòng kiểm tra best_model.keras và class_names.json."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "📸 Chọn ảnh lá cây",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.asarray(image_resized, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    prediction = model.predict(input_tensor, verbose=0)[0]
    pred_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[pred_index]
    confidence = float(prediction[pred_index]) * 100

    info = DISEASE_INFO[predicted_class]

    st.divider()
    st.subheader("🔍 Kết quả")
    st.success(info["name"])
    st.metric("Độ tin cậy của model", f"{confidence:.2f}%")

    st.info(f"**Triệu chứng thường gặp:** {info['symptoms']}")

    st.subheader("💊 Hướng xử lý / thuốc thường gặp")
    st.write(info["treatment"])
    st.caption(info["active"])

    st.warning(
        "⚠️ Đây là gợi ý tham khảo từ model, không phải chẩn đoán chuyên gia. "
        "Thuốc bảo vệ thực vật phải đúng cây + đúng bệnh trên nhãn, đúng liều và thời gian cách ly. "
        "Quy định và sản phẩm được phép dùng có thể khác theo Việt Nam."
    )

    st.subheader("📊 Top 3 dự đoán")
    top_indices = np.argsort(prediction)[-3:][::-1]

    for i in top_indices:
        cls = CLASS_NAMES[int(i)]
        score = float(prediction[int(i)]) * 100
        st.write(f"**{DISEASE_INFO[cls]['name']}** — {score:.2f}%")

st.divider()
st.caption(
    "Model được huấn luyện trên PlantVillage với 15 class. "
    "Ảnh thực địa có thể khác dữ liệu huấn luyện, vì vậy không nên dựa vào AI alone để quyết định phun thuốc."
)
