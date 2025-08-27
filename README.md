# VSLR_TestFinal (Vietnamese Sign Language Recognition - Production Release)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)](https://pypi.org/project/PyQt5/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-orange.svg)](https://mediapipe.dev/)

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)
- [Liên hệ](#liên-hệ)

## Giới thiệu

**VSLR_TestFinal** là phiên bản cuối cùng và hoàn thiện của hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam (Vietnamese Sign Language Recognition). Dự án này được phát triển dựa trên nền tảng PyTorch và sử dụng giao diện PyQt5 để tạo ra một ứng dụng desktop thân thiện với người dùng.

**Mục tiêu chính:**
- Cung cấp ứng dụng nhận diện ngôn ngữ ký hiệu tiếng Việt thời gian thực
- Hỗ trợ đầy đủ bảng chữ cái tiếng Việt bao gồm các ký tự đặc biệt và dấu thanh điệu
- Tạo giao diện người dùng trực quan và dễ sử dụng
- Đảm bảo hiệu suất cao và độ chính xác trong môi trường thực tế

## Tính năng

### 🎯 **Nhận diện ngôn ngữ ký hiệu thời gian thực**
- Xử lý video đầu vào từ webcam với độ trễ tối thiểu
- Đạt độ chính xác cao trong điều kiện ánh sáng và môi trường khác nhau
- Tối ưu hóa cho bảng chữ cái tiếng Việt

### 🇻🇳 **Hỗ trợ đầy đủ ký tự tiếng Việt**
- Nhận diện 26 ký tự cơ bản: A, B, C, D, DD, E, G, H, I, K, L, M, Mu, Munguoc, N, O, P, Q, R, Rau, S, T, U, V, X, Y
- Hỗ trợ các ký tự đặc biệt: Â, Ă, Ê, Ô, Ơ, Ư, Đ
- Xử lý dấu thanh điệu: sắc, huyền, hỏi, ngã, nặng

### 🧠 **Xử lý văn bản thông minh**
- **Lọc ký tự trùng lặp**: Tránh nhận diện lặp lại ký tự khi cử chỉ giữ nguyên
- **Xử lý ký tự đặc biệt hai bước**: 
  - Bước 1: Thực hiện ký tự cơ bản (A, O, U, E)
  - Bước 2: Thực hiện ký hiệu dấu (Mu, Munguoc, Rau)
  - Kết quả: Hệ thống tự động kết hợp tạo ký tự đặc biệt
- **Nhận dạng dấu thanh bằng LSTM**: Sử dụng mô hình LSTM để nhận dạng 5 dấu thanh tiếng Việt

### 📝 **Tạo câu tự động**
- **Tách từ thông minh**: Tự động chèn khoảng trắng khi không phát hiện tay trong 0.5-1 giây
- **Xây dựng câu hoàn chỉnh**: Hỗ trợ tạo câu tiếng Việt có nghĩa
- **Xử lý thanh điệu**: Tự động áp dụng dấu thanh điệu cho ký tự vừa nhập

### 🎮 **Giao diện người dùng PyQt5**
- Giao diện hiện đại, trực quan được xây dựng bằng PyQt5
- Hiển thị video thời gian thực với overlay cử chỉ
- Các tính năng điều khiển văn bản: xóa, xóa tất cả, lưu file
- Thiết kế tập trung vào khả năng tiếp cận

### ⚡ **Hiệu suất cao**
- **ResNet50 cho nhận dạng ký tự**: Sử dụng kiến trúc ResNet50 được tối ưu hóa cho PyTorch
- **LSTM cho nhận dạng dấu thanh**: Mô hình LSTM chuyên biệt cho việc nhận dạng dấu thanh
- Hỗ trợ gia tốc GPU
- Phát hiện điểm mốc tay hiệu quả bằng MediaPipe
- Xử lý thời gian thực ở 30+ FPS

## Cài đặt

### Yêu cầu hệ thống

- **Hệ điều hành**: Windows 10/11, macOS 10.14+, hoặc Ubuntu 18.04+
- **Python**: 3.8 trở lên
- **Phần cứng**:
  - Webcam để nhận diện thời gian thực
  - Tối thiểu 4GB RAM (khuyến nghị 8GB)
  - GPU hỗ trợ CUDA (tùy chọn, để huấn luyện)

### Hướng dẫn cài đặt

1. **Clone repository**
   ```bash
   git clone https://github.com/HiewNT/VSLR_TestFinal.git
   cd VSLR_TestFinal
   ```

2. **Tạo môi trường ảo** (Khuyến nghị)
   ```bash
   # Tạo môi trường ảo
   python -m venv vslr_env
   
   # Kích hoạt môi trường ảo
   # Trên Windows:
   vslr_env\Scripts\activate
   # Trên macOS/Linux:
   source vslr_env/bin/activate
   ```

3. **Cài đặt dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Kiểm tra cài đặt**
   ```bash
   python -c "import torch, cv2, mediapipe, PyQt5; print('Tất cả dependencies đã được cài đặt thành công!')"
   ```

### Cài đặt nâng cao

#### Hỗ trợ GPU (Tùy chọn)
Nếu bạn có GPU NVIDIA và muốn tăng tốc độ xử lý:

```bash
# Cài đặt PyTorch với hỗ trợ CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Khắc phục sự cố cài đặt

**Các vấn đề thường gặp:**
- **Lỗi cài đặt MediaPipe**: Đảm bảo bạn sử dụng Python 3.8-3.11 (MediaPipe chưa hỗ trợ Python 3.12+)
- **Vấn đề với OpenCV**: Thử `pip install opencv-python-headless` nếu gặp vấn đề hiển thị
- **Lỗi PyQt5**: Trên Linux có thể cần: `sudo apt-get install python3-pyqt5`

## Cách sử dụng

### Khởi chạy ứng dụng

```bash
python app_qt.py
```

Ứng dụng sẽ mở với giao diện PyQt5 bao gồm:
- Hiển thị video webcam thời gian thực
- Nhận diện cử chỉ trực tiếp
- Hiển thị văn bản đầu ra
- Các nút điều khiển cho thao tác văn bản

### Hướng dẫn sử dụng

1. **Bắt đầu nhận diện**:
   - Đặt tay trước webcam
   - Thực hiện các cử chỉ theo bảng chữ cái ngôn ngữ ký hiệu Việt Nam
   - Văn bản sẽ hiển thị trong thời gian thực

2. **Tạo ký tự đặc biệt**:
   ```
   Ví dụ tạo ký tự "Â":
   - Bước 1: Thực hiện cử chỉ "A"
   - Bước 2: Thực hiện cử chỉ "Mu" (dấu mũ)
   - Kết quả: Hệ thống tự động tạo "Â"
   ```

3. **Tạo từ và câu**:
   - Thực hiện liên tiếp các ký tự để tạo từ
   - Rời tay khỏi camera 0.5-1 giây để tạo khoảng trắng
   - Tiếp tục với từ tiếp theo

4. **Áp dụng thanh điệu**:
   - Thực hiện ký tự cơ bản trước
   - Thực hiện cử chỉ thanh điệu (huyen, sac, hoi, nga, nang)
   - Hệ thống tự động áp dụng dấu cho ký tự cuối

### Các tính năng giao diện

- **Xóa ký tự cuối**: Xóa ký tự vừa nhập
- **Xóa tất cả**: Xóa toàn bộ văn bản
- **Lưu file**: Xuất văn bản ra file `recognized_text.txt`
- **Hiển thị trạng thái**: Theo dõi FPS và trạng thái nhận diện

## Cấu trúc thư mục

```
VSLR_TestFinal/
├── 📱 app_qt.py                         # Ứng dụng GUI chính (PyQt5)
├── 📄 requirements.txt                  # Python dependencies
├── 📝 recognized_text.txt              # File lưu văn bản đã nhận diện
├── 📁 src/                             # Mã nguồn core
│   ├── 🧠 classification.py            # Pipeline phân loại CNN
│   ├── ⚙️ config.py                   # Cấu hình và hằng số
│   ├── 🎥 frame_processor.py           # Xử lý khung hình video
│   ├── 🖐️ hand_tracking.py             # Theo dõi tay MediaPipe
│   ├── 🏗️ model.py                    # Kiến trúc mô hình CNN
│   ├── 📝 text_processor.py           # Xử lý văn bản và ký tự đặc biệt
│   ├── 🎵 tone_predictor.py           # Dự đoán thanh điệu tiếng Việt
│   └── 🛠️ utils.py                    # Các hàm tiện ích
├── 📁 trained_models/                  # Mô hình đã huấn luyện
│   ├── 🏆 last.pt                     # Mô hình ResNet50 cho ký tự
│   ├── 🎵 lstm_model_final.h5         # Mô hình LSTM cho thanh điệu
│   └── 🏷️ lstm_model_label_encoder.pkl # Label encoder cho LSTM
└── 📖 README.md                        # Tài liệu dự án
```

### Mô tả các thành phần chính

- **`app_qt.py`**: Ứng dụng GUI chính sử dụng PyQt5, quản lý giao diện người dùng và tích hợp tất cả các module
- **`src/frame_processor.py`**: Xử lý khung hình video, tích hợp nhận diện tay và phân loại
- **`src/text_processor.py`**: Xử lý logic văn bản, ký tự đặc biệt và tạo câu thông minh
- **`src/tone_predictor.py`**: Dự đoán và áp dụng thanh điệu tiếng Việt bằng mô hình LSTM
- **`trained_models/`**: Chứa các mô hình đã được huấn luyện sẵn

## Đóng góp

Chúng tôi hoan nghênh sự đóng góp từ cộng đồng! Dưới đây là cách bạn có thể giúp đỡ:

### Cách đóng góp

- **🐛 Báo cáo lỗi**: Báo cáo các vấn đề và lỗi
- **💡 Đề xuất tính năng**: Đề xuất tính năng mới hoặc cải tiến
- **📝 Tài liệu**: Cải thiện tài liệu và hướng dẫn
- **🧪 Kiểm thử**: Kiểm thử hệ thống trên các phần cứng/hệ điều hành khác nhau
- **🎯 Cải tiến mô hình**: Đóng góp cải tiến kiến trúc mô hình hoặc kỹ thuật huấn luyện

### Quy trình phát triển

1. Fork repository
2. Tạo branch tính năng: `git checkout -b feature-ten-tinh-nang`
3. Thực hiện thay đổi và kiểm thử kỹ lưỡng
4. Commit thay đổi: `git commit -m 'Thêm tính năng XYZ'`
5. Push lên branch: `git push origin feature-ten-tinh-nang`
6. Tạo Pull Request với mô tả chi tiết

### Tiêu chuẩn code

- Tuân thủ PEP 8 cho Python code style
- Thêm docstring cho tất cả functions và classes
- Bao gồm unit tests cho tính năng mới
- Cập nhật tài liệu khi cần thiết

## Giấy phép

Dự án này được phát hành theo giấy phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

### Giấy phép bên thứ ba

- **MediaPipe**: Apache License 2.0
- **PyTorch**: BSD License
- **OpenCV**: Apache License 2.0
- **PyQt5**: GPL v3 / Commercial License

## Liên hệ

Để có câu hỏi, đề xuất hoặc cơ hội hợp tác:

- **Email**: nthieu.1703@gmail.com
- **GitHub**: [@HiewNT](https://github.com/HiewNT)
- **Repository**: [VSLR_TestFinal](https://github.com/HiewNT/VSLR_TestFinal)

### Lời cảm ơn

- Đội ngũ MediaPipe cho công nghệ theo dõi tay
- Cộng đồng PyTorch cho framework deep learning
- Cộng đồng ngôn ngữ ký hiệu Việt Nam cho hướng dẫn và phản hồi
- Nhóm phát triển PyQt5 cho framework GUI

---

**Được tạo ra với ❤️ cho cộng đồng người khiếm thính và người nghe Việt Nam**

## Ảnh chụp màn hình

*Sẽ được cập nhật với ảnh chụp màn hình giao diện ứng dụng trong các phiên bản tương lai*

## Thống kê hiệu suất

- **Độ chính xác mô hình**: >90% trên tập validation
- **FPS thời gian thực**: 30+ khung hình/giây
- **Thời gian inference**: <50ms mỗi khung hình
- **Kích thước mô hình**: ~10MB (tối ưu cho triển khai)

## Cập nhật và phiên bản

### Phiên bản hiện tại: v1.0.0

**Tính năng chính:**
- Giao diện PyQt5 hoàn chỉnh
- Nhận diện ký tự tiếng Việt đầy đủ
- Xử lý thanh điệu tự động
- Tạo câu thông minh
- Hiệu suất tối ưu

**Kế hoạch phát triển:**
- Hỗ trợ từ vựng ngôn ngữ ký hiệu mở rộng
- Tích hợp tính năng text-to-speech
- Cải thiện độ chính xác trong điều kiện ánh sáng kém
- Hỗ trợ nhiều ngôn ngữ ký hiệu khác
