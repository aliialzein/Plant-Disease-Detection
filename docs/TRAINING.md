# Huấn luyện lại MobileNetV2

Notebook: notebooks/train_mobilenetv2_colab.ipynb

## Chuẩn bị dữ liệu

Đặt PlantVillage theo cấu trúc gồm 15 thư mục lớp. Script hỗ trợ cả biến thể tên thư mục Tomato có một hoặc hai dấu gạch dưới. Cần đủ đúng 15 lớp; nếu thiếu lớp, chương trình dừng để tránh gán nhãn sai.

## Chạy trên Colab

1. Mở notebook bằng nút Open in Colab.
2. Chọn DATA_SOURCE trong notebook: drive_folder (mặc định MyDrive/PlantVillage) hoặc upload_zip để tải một file ZIP từ máy tính lên Colab.
3. Chạy lần lượt các cell. Nếu dùng Drive và dataset ở nơi khác, sửa DATASET_DIR.
4. Notebook lưu ZIP kết quả vào MyDrive/plant_disease_training_artifacts.zip.

## Quy trình

- Chia dữ liệu có phân tầng: 70% train, 15% validation, 15% test; seed mặc định 42.
- Augmentation chỉ áp dụng trong lúc train.
- Input ảnh trong pipeline là [0,1]; model có lớp Rescaling chuyển sang [-1,1] theo yêu cầu MobileNetV2.
- Phase 1 huấn luyện lớp phân loại với backbone đóng băng.
- Phase 2 fine-tune 30 lớp cuối với learning rate thấp; Batch Normalization giữ đóng băng.
- Chọn checkpoint theo validation accuracy và early stopping theo validation loss.
- Chỉ dùng tập test sau khi đã chọn checkpoint tốt nhất.
- Xuất classification report, confusion matrix, learning curves, metadata và model.

## Đưa model vào ứng dụng

Giải nén ZIP và chép best_model.keras cùng class_names.json vào thư mục model/. Ứng dụng ưu tiên model .keras, đọc thứ tự lớp từ JSON và kiểm tra số đầu ra trước khi dự đoán. Model .h5 cũ vẫn được giữ làm phương án dự phòng nếu chưa có model mới.

Không commit dataset hoặc model đã huấn luyện lên GitHub. Lưu model artifacts trên Google Drive.

## Giới hạn

- Đây là đánh giá nội bộ trên PlantVillage; cần ảnh thực địa độc lập để đánh giá khả năng tổng quát hóa.
- Confidence không phải xác suất chẩn đoán chắc chắn.
- Hướng dẫn thuốc trong ứng dụng là nội dung tham khảo tĩnh, không phải kê đơn.
