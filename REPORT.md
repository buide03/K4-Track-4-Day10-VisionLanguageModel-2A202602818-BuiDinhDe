Họ tên: Bùi Đình Đề · MSSV: 2A202602818
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | | | | | | | | |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
<điền>

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
<điền>

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
<điền>

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
<điền>

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
<điền>

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Số liệu tôi đo được sau `prepare_data.py` (`dataset_version` d10-21e4070301):

| Split | Ảnh | Phiên quay | Box | pothole | crack | manhole | Ảnh không box |
|---|---|---|---|---|---|---|---|
| train | 1422 | 13 | 3515 | 1074 | 1750 | 691 | 0 |
| val | 276 | 2 | 639 | 66 | 470 | 103 | 0 |
| test | 311 | 3 | 583 | 121 | 299 | 163 | 0 |

Kiểm tra rò rỉ: số phiên chung giữa train–val, train–test, val–test đều bằng 0; không `image_id` nào xuất hiện ở hai split.

**Vì sao chia ngẫu nhiên cho F1 cao giả.** 2009 ảnh chỉ đến từ 18 phiên quay, trung bình khoảng 110 frame mỗi phiên, cắt từ cùng một video. Hai frame liền nhau cách nhau vài giây nên gần như trùng: cùng con đường, cùng ánh sáng, cùng ổ gà, chỉ dịch vài pixel. Ví dụ hai ảnh val `vlcsnap-2025-02-18-18h30m26s925` và `18h32m52s770` cùng phiên `vlc-2025-02-18-18`, cùng đoạn đường. Nếu chia ngẫu nhiên, gần như mỗi ảnh test đều có một "bản sao" trong train. Model chỉ cần nhớ vị trí box của cảnh đó là đạt IoU ≥ 0,5, nên F1 đo được là khả năng **nhớ**, không phải khả năng **khái quát** sang con đường mới. Khi dùng thật, mọi ảnh đều từ con đường chưa thấy, nên F1 thật sẽ thấp hơn nhiều so với con số báo cáo. Val cũng bị rò rỉ theo, nên mọi lựa chọn dựa trên val (số bước, rank) đều bị chọn lệch theo hướng thuộc lòng.

**Cái giá của việc chia theo phiên.** Không kiểm soát được phân bố lớp. Val chỉ có 66 pothole so với 470 crack, nên tôi chỉ dùng val để gỡ lỗi, không dùng để so sánh setup. Thêm nữa, test chỉ gồm **3 phiên**, nên 311 ảnh không phải 311 mẫu độc lập: các frame trong cùng phiên tương quan mạnh. Vì vậy CI95 tính theo từng ảnh có thể hẹp hơn độ bất định thật (xem Câu 8).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
<điền>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
<điền>

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>
