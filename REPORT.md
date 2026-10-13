Họ tên: Bùi Đình Đề · MSSV: 2A202602818
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — (suy luận 4,29) | — (suy luận 13,7) | 0,086 | 0,060–0,114 | 1,06 | 0,124 | 0,026 | 0,193 |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
<điền>

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

**`parse_valid_rate` = 0,929**: 289/311 output đọc được, 22 ảnh JSON hỏng. Model gốc tuân thủ định dạng khá tốt: không bọc ```json, không có chữ thừa, không có nhãn ngoài 3 lớp. Nhưng nội dung thì sai: F1 = **0,086** (CI95 0,060–0,114), P = 0,111, R = 0,070. Model dự đoán 370 box, chỉ **41 TP** so với 583 box thật (329 FP, 542 FN).

Tôi đã loại trừ lỗi **sai thang**: toạ độ dự đoán nằm trong x ∈ [0, 1000], y ∈ [276, 1000] (y vượt 360, nên không phải pixel). Chấm lại với `--coord pixel` thì F1 rơi về 0,000, còn với `norm1000` là 0,086. Vậy model gốc **đã dùng đúng thang 0–1000**.

**Lỗi 1: bỏ sót vật thể (recall 0,070).** 110/311 ảnh có ít box hơn nhãn, 5 ảnh trả `[]` dù có hư hại. Thường model chỉ trả **một box** cho cả ảnh. Ví dụ `vlcsnap-00200`:
```
GT : [{"label":"crack","bbox_2d":[352,642,534,728]},{"label":"manhole","bbox_2d":[894,756,1000,847]}]
raw: [{"label": "crack", "bbox_2d": [750, 638, 850, 684]}]
```
Nắp cống bị bỏ qua hoàn toàn, còn box crack nằm sai chỗ (IoU = 0).

**Lỗi 2: box sai kích thước so với quy ước gán nhãn (IoU < 0,5 dù tìm đúng chỗ).** Lỗi này có hai dạng ngược nhau:
- *Crack: box khổng lồ phủ cả mặt đường.* `vlcsnap-00203` dự đoán `[0, 476, 1000, 999]` (rộng 1000 × cao 523) cho một vết nứt có nhãn `[322,631,759,775]` (437 × 144), IoU = 0,12. Crack chiếm 244/370 box dự đoán nhưng F1 crack chỉ 0,026: model coi "mặt đường nứt" là một vùng lớn, còn người gán nhãn chia vết nứt thành từng đoạn.
- *Manhole: đúng vị trí nhưng box quá dẹt.* `vlcsnap-00202` dự đoán `[309, 684, 399, 714]`, nhãn là `[275,672,405,722]`: chiều cao 30 so với 50, IoU = **0,415**. Ở `vlcsnap-00204`, chiều cao 30 so với 56, IoU = 0,347. Model nhận ra nắp cống (F1 manhole 0,193, cao nhất trong 3 lớp) nhưng vẽ box sát hơn quy ước của bộ nhãn, nên trượt ngưỡng IoU 0,5.

**Về 22 JSON hỏng.** Cả 22 đều **bị cắt giữa chừng** (chưa đóng `]`). Ô phân tích cũng đếm được 20 ảnh **lặp lại cùng một box**. Model rơi vào vòng lặp sinh mãi một box cho đến khi chạm `max_new_tokens` = 256. Trong khi đó, đáp án đúng dài nhất trên test chỉ 196 token, nên đây là lỗi của model chứ không phải do giới hạn quá chặt.

**Kết luận.** Model gốc *biết* hư hại nằm ở nửa dưới ảnh (y ≥ 276) và nhận ra được nắp cống. Nhưng nó không biết **quy ước của bộ nhãn này**: liệt kê từng vật thể, chia vết nứt thành đoạn, box ôm rộng bao nhiêu. Đó đúng là thứ finetune cần dạy. Mốc để so sánh: E1 phải đạt F1 ≥ 0,186.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

**Kiểm chứng.** Tôi tạo "đáp án hoàn hảo" từ chính nhãn test (311 ảnh, 583 box) theo hai thang, rồi chấm bằng `scripts/score.py` (hàm `evaluate`) với cả hai giá trị `--coord`:

| Nhãn sinh theo | Chấm với `--coord` | F1 | mean IoU | FP/ảnh |
|---|---|---|---|---|
| norm1000 | norm1000 | **1,000** | 0,988 | 0,00 |
| pixel | pixel | **1,000** | 1,000 | 0,00 |
| norm1000 | pixel | **0,000** | 0,000 | 1,87 |
| pixel | norm1000 | **0,000** | 0,000 | 1,87 |

Cùng một bộ box đúng tuyệt đối, chỉ cần đọc sai thang là F1 rơi từ 1,0 xuống **0**. Cả 583 box thành FP (583/311 = 1,87 FP/ảnh), và cả 583 box thật thành FN.

**Vì sao.** Khi đáp án ở thang 0–1000 bị đọc như pixel, trọng tài chia x cho 640 và y cho 360 rồi nhân 1000. Kết quả là box bị **kéo giãn và đẩy xa gốc toạ độ**: ×1,56 theo chiều ngang, ×2,78 theo chiều dọc. Box thật ở giữa ảnh, khoảng y = 500, bị đẩy ra ngoài khung (y ≈ 1390). Box dự đoán và box thật không còn chồng nhau, IoU = 0 < 0,5, nên không có TP nào. Chiều ngược lại cũng vậy: box bị co về phía góc trên trái. Lỗi này nguy hiểm vì JSON vẫn **hợp lệ** (`parse_valid_rate` = 1), chỉ F1 sụp. Nhìn riêng độ hợp lệ của output sẽ không phát hiện ra.

**Round-trip.** Tôi đổi nhãn YOLO → JSON 0–1000 → `parse_output` của trọng tài → pixel trên toàn bộ 4.737 box. Sai số lớn nhất là **0,32 px**, đúng bằng sai số làm tròn lý thuyết 640/1000/2. Vẽ chồng lên ảnh (`report/figures/cp2_roundtrip.jpg`) thấy box quy ngược khít với box gốc. Mean IoU 0,988 (thay vì 1,0) cho thấy nhãn gốc được vẽ trên lưới pixel nguyên, và làm tròn sang lưới 0–1000 làm lệch nhẹ các box nhỏ. Mức lệch này vẫn xa ngưỡng 0,5.

**Vì sao Qwen3-VL dùng 0–1000 chứ không dùng pixel.** Qwen3-VL đổi kích thước ảnh động trước khi đưa vào ViT (patch 16 px, gộp 2×2), nên "pixel" mà model nhìn thấy phụ thuộc độ phân giải đầu vào. Toạ độ tương đối 0–1000 thì không đổi dù ảnh 640×360 hay 1920×1080. Ngoài ra, dữ liệu grounding lúc tiền huấn luyện dùng thang này, nên giữ đúng thang là tận dụng được kiến thức sẵn có của model thay vì bắt model học lại.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

**Cách tôi mask.** Mỗi ví dụ là một hội thoại `user: [ảnh] + PROMPT` → `assistant: JSON`. Tôi mã hoá hai lần bằng `apply_chat_template`: một lần chỉ có phần user (kèm `add_generation_prompt=True`), một lần có cả câu trả lời. Sau đó tôi kiểm tra chuỗi đầu là **tiền tố** của chuỗi sau, rồi gán `labels = -100` cho toàn bộ tiền tố. Ký tự `\n` sau `<|im_end|>` cũng được gán -100, vì model dừng ngay ở `<|im_end|>`. Phần đệm trong batch cũng có `labels = -100`.

**Kiểm chứng trên dữ liệu của tôi:**
- Ảnh 640×360 cho đúng **220** token `<|image_pad|>`.
- Giải mã riêng các token có `labels != -100` ra đúng `[{"label":"crack","bbox_2d":[430,467,583,506]},…]<|im_end|>`, khớp từng ký tự với `objects_to_answer` cộng `<|im_end|>`.
- Ví dụ đầu tiên dài 609 token: prompt 359 (220 ảnh + 139 chữ), câu trả lời 249, 1 token đuôi.

**Tỉ lệ token.** Trên 1.422 ảnh train, prompt **luôn dài 359 token** (cùng kích thước ảnh, cùng câu hỏi). Câu trả lời trung bình chỉ **70 token** (dài nhất 366), chuỗi trung bình 430 token. Như vậy loss chỉ tính trên trung bình **16,3%** số token. Nếu không mask, khoảng **84%** số token đóng góp vào loss là những thứ không cần học.

**Vì sao chỉ tính loss trên câu trả lời.** Khi suy luận, model chỉ *sinh* phần sau `<|im_start|>assistant`. Prompt và ảnh luôn là *đầu vào* có sẵn, không bao giờ phải sinh. Vì vậy mục tiêu học phải là P(câu trả lời | ảnh, prompt), không phải xác suất của cả chuỗi.

**Nếu tính loss trên cả prompt và token ảnh thì sao:**
1. *Tín hiệu bị pha loãng.* Loss là trung bình trên mọi token có nhãn. Khi 84% token là prompt, phần trả lời chỉ chiếm khoảng 16% trọng số của gradient. Cùng lr và 300 bước, model học nhiệm vụ thật chậm hơn nhiều.
2. *Loss "giảm đẹp" nhưng vô nghĩa.* 139 token chữ của prompt giống hệt nhau ở mọi ví dụ, nên model thuộc lòng rất nhanh. Còn 220 token ảnh thì nhãn luôn là `<|image_pad|>`, đoán đúng là việc tầm thường. Đường loss giảm nhanh, tạo cảm giác model đang học, trong khi F1 grounding hầu như không đổi. Đây là kiểu lỗi đánh lừa người theo dõi loss.
3. *Học sai hành vi.* Model bị dạy để "viết tiếp" phần câu hỏi và chuỗi `<|image_pad|>`, vốn là các vị trí mà đặc trưng ảnh được chèn vào thay cho embedding chữ. Điều này làm nhiễu đúng phần LLM đọc ảnh, và có thể khiến model sinh ra văn bản kiểu prompt thay vì JSON.

**Vì sao phải giữ `<|im_end|>` trong phần có loss.** Ở E0, 22/311 output bị cắt vì model lặp một box đến khi chạm 256 token. Học token `<|im_end|>` ngay sau dấu `]` là cách dạy model **dừng đúng lúc**. Đây là điều tôi kỳ vọng thấy qua `parse_valid_rate` của E1 (rubric yêu cầu ≥ 0,95).

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
