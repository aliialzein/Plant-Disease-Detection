# 🌱 Plant Disease AI

AI nhận diện bệnh cây trồng từ hình ảnh lá, sử dụng mô hình **MobileNetV2** đã được huấn luyện trước trên tập dữ liệu PlantVillage.

Ứng dụng có giao diện **Streamlit**, cho phép người dùng tải ảnh lá lên và nhận kết quả dự đoán bệnh cùng hướng dẫn xử lý/tham khảo.

---

## ✨ Features

* 📷 Upload ảnh lá cây
* 🤖 Nhận diện bệnh bằng MobileNetV2
* 📊 Hiển thị độ tin cậy của dự đoán
* 🔝 Hiển thị **Top-3 predictions**
* 🌱 Hỗ trợ 15 nhóm bệnh/tình trạng cây
* 💊 Hiển thị hướng xử lý và hoạt chất phổ biến để tham khảo
* 🇻🇳 Tên bệnh và hướng dẫn bằng tiếng Việt
* ⚠️ Cảnh báo an toàn khi sử dụng thuốc bảo vệ thực vật
* 🌐 Web UI bằng Streamlit

---

## 🧠 Model

Model sử dụng:

* **MobileNetV2**
* Pretrained weights: **ImageNet**
* Input size: **128 × 128**
* Output: **15 classes**
* Global Average Pooling
* Dropout: `0.3`
* Fine-tuning các layer cuối của MobileNetV2

Model được lưu tại:

```text
model/best_model.h5
```

---

## 🌿 Supported Diseases

### 🫑 Pepper / Ớt

#### 1. Đốm vi khuẩn trên ớt

**Class:** `Pepper__bell___Bacterial_spot`

**Triệu chứng:**

* Đốm nhỏ màu nâu hoặc đen trên lá
* Có thể xuất hiện quầng vàng
* Bệnh có thể lan rộng khi lá thường xuyên ẩm

**Xử lý tham khảo:**

* Loại bỏ lá bị bệnh nặng
* Hạn chế tưới nước lên lá
* Tăng thông thoáng
* Vệ sinh dụng cụ và khu vực trồng
* Một số sản phẩm chứa hợp chất **đồng (copper)** có thể được sử dụng tùy theo nhãn thuốc và quy định địa phương.

---

#### 2. Ớt khỏe mạnh

**Class:** `Pepper__bell___healthy`

Không phát hiện dấu hiệu rõ ràng của các bệnh nằm trong phạm vi model.

---

# 🥔 Potato / Khoai tây

### 3. Bệnh cháy lá sớm

**Class:** `Potato___Early_blight`

**Triệu chứng:**

* Đốm nâu trên lá
* Thường có các vòng tròn đồng tâm giống hình bia
* Xuất hiện nhiều ở các lá già

**Xử lý tham khảo:**

* Cắt bỏ lá bị bệnh nặng
* Giữ lá khô
* Tăng thông thoáng
* Vệ sinh tàn dư cây bệnh
* Một số thuốc diệt nấm chứa **chlorothalonil** thường được sử dụng trong quản lý bệnh, tùy đăng ký và nhãn sản phẩm.

---

### 4. Bệnh mốc sương / cháy lá muộn

**Class:** `Potato___Late_blight`

**Triệu chứng:**

* Vết bệnh màu nâu đậm hoặc đen
* Lan rất nhanh trong điều kiện ẩm
* Có thể xuất hiện lớp mốc trắng ở mặt dưới lá

**Xử lý tham khảo:**

* Loại bỏ phần cây bị nhiễm nặng
* Giảm độ ẩm và tăng thông thoáng
* Không tưới nước trực tiếp lên tán lá
* Sử dụng sản phẩm được đăng ký để quản lý **Phytophthora / late blight** nếu phù hợp.

---

### 5. Khoai tây khỏe mạnh

**Class:** `Potato___healthy`

Không phát hiện dấu hiệu rõ ràng của các bệnh nằm trong phạm vi model.

---

# 🍅 Tomato / Cà chua

### 6. Đốm vi khuẩn

**Class:** `Tomato__Bacterial_spot`

**Triệu chứng:**

* Đốm nhỏ màu nâu/đen
* Có thể có quầng vàng
* Bệnh phát triển mạnh khi lá ẩm

**Xử lý tham khảo:**

* Loại bỏ lá bệnh
* Giữ tán lá khô
* Tăng thông thoáng
* Một số sản phẩm chứa **copper** có thể được sử dụng tùy nhãn thuốc.

---

### 7. Cháy lá sớm

**Class:** `Tomato__Early_blight`

**Triệu chứng:**

* Đốm nâu trên lá
* Vòng đồng tâm giống hình bia
* Thường bắt đầu từ các lá già

**Xử lý tham khảo:**

* Cắt bỏ lá bị bệnh
* Vệ sinh tàn dư cây
* Hạn chế nước đọng trên lá
* Tăng thông thoáng
* **Chlorothalonil** là một hoạt chất thường được sử dụng trong quản lý bệnh này ở một số nơi, tùy đăng ký.

---

### 8. Mốc sương / Cháy lá muộn

**Class:** `Tomato__Late_blight`

**Triệu chứng:**

* Vết bệnh lớn màu nâu
* Lan nhanh
* Có thể xuất hiện mốc trắng khi độ ẩm cao

**Xử lý tham khảo:**

* Loại bỏ cây hoặc phần cây nhiễm nặng
* Giảm độ ẩm
* Tăng thông thoáng
* Sử dụng thuốc được đăng ký cho **Phytophthora / late blight** theo đúng nhãn.

---

### 9. Mốc lá

**Class:** `Tomato__Leaf_Mold`

**Triệu chứng:**

* Mặt trên lá có vùng vàng
* Mặt dưới có lớp mốc màu xanh ô-liu hoặc nâu
* Thường xuất hiện trong điều kiện ẩm

**Xử lý tham khảo:**

* Giảm độ ẩm
* Tăng thông gió
* Hạn chế tưới lên lá
* Vệ sinh lá bệnh
* Một số sản phẩm chứa **copper hydroxide** có thể được sử dụng tùy đăng ký và nhãn thuốc.

---

### 10. Đốm lá Septoria

**Class:** `Tomato__Septoria_leaf_spot`

**Triệu chứng:**

* Nhiều đốm tròn nhỏ
* Tâm đốm màu xám/nâu nhạt
* Viền đậm
* Có thể thấy các chấm đen nhỏ trong vết bệnh

**Xử lý tham khảo:**

* Loại bỏ lá bệnh
* Giữ tán lá khô
* Tăng thông thoáng
* Vệ sinh tàn dư cây
* **Chlorothalonil** thường được sử dụng trong quản lý bệnh ở một số khu vực, tùy đăng ký và nhãn thuốc.

---

### 11. Nhện đỏ hai chấm

**Class:** `Tomato__Spider_mites_Two_spotted_spider_mite`

**Triệu chứng:**

* Các chấm vàng li ti trên lá
* Lá có thể bạc màu
* Có thể nhìn thấy mạng tơ mảnh
* Thường tập trung ở mặt dưới lá

**Xử lý tham khảo:**

* Kiểm tra kỹ mặt dưới lá
* Dùng nước phun nhẹ để giảm mật độ nhện
* Bảo vệ thiên địch
* Một số sản phẩm như **xà phòng diệt côn trùng / potassium salts of fatty acids** hoặc **horticultural oil** có thể được sử dụng nếu phù hợp với cây và nhãn thuốc.

---

### 12. Đốm lá Target Spot

**Class:** `Tomato__Target_Spot`

**Triệu chứng:**

* Đốm bệnh có các vòng đồng tâm
* Hình dạng giống mục tiêu
* Có thể làm lá vàng và rụng sớm

**Xử lý tham khảo:**

* Loại bỏ lá bệnh
* Tăng thông thoáng
* Giữ tán lá khô
* Vệ sinh tàn dư cây
* Có thể sử dụng thuốc diệt nấm đã được đăng ký cho bệnh phù hợp.

---

### 13. Virus xoăn vàng lá cà chua

**Class:** `Tomato__Tomato_YellowLeaf__Curl_Virus`

**Triệu chứng:**

* Lá nhỏ
* Vàng lá
* Lá cuộn hoặc cong
* Cây có thể sinh trưởng kém

**Xử lý tham khảo:**

* Không có thuốc diệt virus trực tiếp
* Loại bỏ cây nhiễm nặng
* Kiểm soát côn trùng môi giới truyền virus
* Sử dụng cây giống sạch bệnh
* Vệ sinh khu vực trồng.

---

### 14. Virus khảm cà chua

**Class:** `Tomato__Tomato_mosaic_virus`

**Triệu chứng:**

* Lá có các vùng xanh đậm/xanh nhạt xen kẽ
* Lá có thể biến dạng
* Cây sinh trưởng kém

**Xử lý tham khảo:**

* Không có thuốc diệt virus trực tiếp
* Loại bỏ cây nhiễm bệnh
* Khử trùng dụng cụ
* Rửa tay sau khi tiếp xúc với cây bệnh
* Sử dụng hạt giống/cây giống sạch bệnh.

---

### 15. Cà chua khỏe mạnh

**Class:** `Tomato__healthy`

Không phát hiện dấu hiệu rõ ràng của các bệnh nằm trong phạm vi model.

---

# 📊 Prediction

Ứng dụng hiển thị:

```text
Top 1
Bệnh cháy lá sớm
Confidence: 92.4%

Top 2
Đốm lá Septoria
Confidence: 4.8%

Top 3
Đốm Target Spot
Confidence: 1.7%
```

Confidence là mức độ mô hình tự tin đối với ảnh đầu vào, **không phải xác suất chẩn đoán chính xác ngoài thực tế**.

---

# 💊 Treatment Guide

Thông tin thuốc trong ứng dụng chỉ nhằm **tham khảo giáo dục**.

Không nên hiểu kết quả AI là đơn thuốc.

Trước khi sử dụng bất kỳ thuốc bảo vệ thực vật nào:

1. Kiểm tra đúng **loại cây**.
2. Kiểm tra đúng **bệnh**.
3. Đọc kỹ **nhãn sản phẩm**.
4. Sử dụng đúng liều lượng.
5. Tuân thủ **thời gian cách ly (PHI)**.
6. Tuân thủ quy định đăng ký thuốc tại địa phương.
7. Không tự ý tăng liều hoặc trộn nhiều thuốc nếu nhãn không cho phép.
8. Khi bệnh nghiêm trọng, nên tham khảo cán bộ kỹ thuật/nông nghiệp địa phương.

---

# ⚠️ Important Limitation

Model được huấn luyện trên dữ liệu PlantVillage và chỉ nhận diện các lớp bệnh nằm trong dataset.

Vì vậy:

* Không phải mọi bệnh cây đều được nhận diện.
* Ảnh thực tế ngoài đồng có thể khác đáng kể ảnh training.
* Ánh sáng, góc chụp, nền ảnh và chất lượng ảnh có thể ảnh hưởng kết quả.
* Confidence cao **không đồng nghĩa với chẩn đoán chắc chắn**.
* Không nên sử dụng AI làm căn cứ duy nhất để quyết định phun thuốc.

---

# 🚀 Run Locally

Clone repository:

```bash
git clone https://github.com/ShimeKano/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

Cài dependencies:

```bash
pip install -r requirments.txt
```

Chạy Streamlit:

```bash
streamlit run app.py
```

Sau đó mở địa chỉ Streamlit hiển thị trong terminal.

---

# 📁 Project Structure

```text
Plant-Disease-Detection/
│
├── app.py
├── requirments.txt
├── README.md
│
├── model/
│   └── best_model.h5
│
└── notebooks/
    └── training notebook
```

---

# 🛠️ Technology

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy
* Streamlit
* PlantVillage Dataset

---

# 📌 Dataset

Model được xây dựng dựa trên dữ liệu PlantVillage.

Dataset tập trung vào hình ảnh lá cây trong điều kiện tương đối có kiểm soát, vì vậy hiệu năng trên ảnh thực tế ngoài đồng có thể khác so với kết quả trên dataset.

---

# 🔮 Future Improvements

Các hướng phát triển tiếp theo:

* [ ] Hỗ trợ nhiều loại cây hơn
* [ ] Sử dụng model input resolution cao hơn
* [ ] Thu thập ảnh thực tế tại Việt Nam
* [ ] Object Detection để tìm lá bệnh trong ảnh toàn cây
* [ ] Segmentation vùng bệnh
* [ ] Grad-CAM / Explainable AI thực sự
* [ ] Theo dõi lịch sử chẩn đoán
* [ ] Database bệnh cây
* [ ] Gợi ý xử lý theo điều kiện thời tiết
* [ ] Hỗ trợ tiếng Anh
* [ ] Mobile App
* [ ] API cho hệ thống khác

---

# ⚠️ Disclaimer

Đây là một **prototype AI nghiên cứu/giáo dục**, không phải hệ thống chẩn đoán nông nghiệp chuyên nghiệp.

Kết quả AI chỉ nên được sử dụng như **thông tin tham khảo**.

Thông tin về thuốc bảo vệ thực vật trong README/app không thay thế hướng dẫn của nhà sản xuất, chuyên gia nông nghiệp hoặc quy định pháp luật địa phương.

Luôn đọc nhãn thuốc và tuân thủ liều lượng, thời gian cách ly và quy định sử dụng trước khi áp dụng thực tế.


## 🔁 Retrain the model on Google Colab

A reproducible MobileNetV2 training pipeline is available:

- [Open the training notebook in Google Colab](https://colab.research.google.com/github/ShimeKano/Plant-Disease-Detection/blob/feature/reproducible-mobilenetv2-colab-training/notebooks/train_mobilenetv2_colab.ipynb)
- [Training workflow and dataset layout](docs/TRAINING.md)
- Training implementation: `src/train.py`

The notebook expects a PlantVillage dataset containing the same 15 supported classes, stored in Google Drive (default: `MyDrive/PlantVillage`). It creates stratified train/validation/test splits, applies augmentation only during training, fine-tunes MobileNetV2, and exports a test-set classification report and confusion matrix. The trained model and class mapping are saved to Google Drive; they are not committed to this repository.

After training, copy `best_model.keras` and `class_names.json` into `model/`. The app prioritizes this model and checks that its output dimension matches the class list. The existing `best_model.h5` remains as a fallback.

**Important:** training metrics are not known until you execute the notebook. Do not treat example or previous model metrics as results of this pipeline.
