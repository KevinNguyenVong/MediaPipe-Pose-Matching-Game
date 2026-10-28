# MediaPipe Pose Matching Game

Ứng dụng trò chơi tương tác thời gian thực qua camera sử dụng **Python**, **OpenCV** và **MediaPipe Pose**. Hệ thống trích xuất 33 điểm đặc trưng cơ thể (pose landmarks), tính toán góc khớp xương bằng lượng giác và đối chiếu với 6 tư thế mẫu để chấm điểm độ tương đồng.

## 1. Danh sách các thư viện cần thiết
Các gói thư viện cần thiết được khai báo trong tệp `requirements.txt`:
- `opencv-python>=4.8.0`
- `mediapipe==0.10.14`
- `numpy<2.0.0`
- `sounddevice>=0.4.6`

## 2. Hướng dẫn cài đặt và chạy chương trình

### Yêu cầu hệ thống
- Python 3.10 hoặc Python 3.11.
- Máy tính có kết nối Webcam.

### Các bước cài đặt
1. **Tải mã nguồn (Clone repository):**
   ```bash
   git clone [https://github.com/](https://github.com/)<ten-tai-khoan-github-cua-ban>/MediaPipe-Pose-Matching-Game.git
   cd MediaPipe-Pose-Matching-Game
   ```

2. **Tạo và kích hoạt môi trường ảo (Virtual Environment):**
   ```bash
   py -3.11 -m venv venv
   venv\Scripts\activate
   ```

3. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng:**
   ```bash
   python main.py
   ```

## 3. Hướng dẫn sử dụng ứng dụng
1. Đứng cách camera từ **1,5 m đến 2,5 m** để các khớp mục tiêu hiển thị rõ trong khung hình.
2. Quan sát mô hình khung xương mẫu (**TARGET POSE**) ở góc trên bên phải màn hình và hướng dẫn trên thanh tiêu đề.
3. Thực hiện tư thế theo mẫu:
   - **Vòng tròn màu đỏ tại khớp:** Sai số góc lớn hơn 25°, cần điều chỉnh lại vị trí tay hoặc chân.
   - **Vòng tròn màu xanh lá tại khớp:** Góc khớp đã đạt yêu cầu (sai số ≤ 25°).
4. Khi độ tương đồng (**SIMILARITY**) đạt từ **80% trở lên**, giữ nguyên tư thế trong khoảng **1,5 giây** (35 khung hình) để nạp đầy thanh **HOLD**, nhận điểm thưởng và chuyển sang màn tiếp theo.
5. **Các phím tắt điều khiển:**
   - Phím `H`: Bật / Tắt bảng hướng dẫn sử dụng.
   - Phím `N`: Chuyển nhanh sang tư thế tiếp theo.
   - Phím `R`: Đặt lại trò chơi từ màn đầu tiên.
   - Phím `Q`: Đóng ứng dụng.

## 4. Video Demo
- **Liên kết Video Demo:** (https://drive.google.com/file/d/1GSywKLdHjb5n2MXBpTW2Nbs1uiKO7YGx/view?usp=sharing)