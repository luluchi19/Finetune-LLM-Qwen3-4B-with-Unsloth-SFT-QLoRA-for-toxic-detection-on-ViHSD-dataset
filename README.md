# Qwen3 cho nhận diện nội dung độc hại tiếng Việt

Tinh chỉnh **Qwen3-4B-Instruct-2507** bằng **SFT + QLoRA** để phân loại bình luận mạng xã hội tiếng Việt, đồng thời sinh lời giải thích và trích dẫn dấu hiệu vi phạm dưới dạng JSON.

**Kết quả thực nghiệm:** điểm Macro F1 được notebook báo cáo tăng từ **0,3721 → 0,6207** trên **6.680 mẫu test**, sau một epoch huấn luyện và chỉ cập nhật khoảng **0,81% tham số**. Cách tính điểm và phạm vi của kết luận được trình bày ở phần kết quả bên dưới.

[Notebook thực nghiệm](./Fine_tune_LLM_Unsloth_SFT_QLoRA_Qwen3_4B_phân_biệt_câu_nói_toxic_trong_ViHSD.ipynb) · [Dataset](https://huggingface.co/datasets/vominhmanh/vihsd-explainable) · [W&amp;B run](https://forge.coreweave.com/wandb/giangnguyeen1910-luluchi19/qwen3-vihsd/runs/n9qhscs3)

## Bài toán và dữ liệu

Bình luận tiếng Việt trên mạng xã hội thường có tiếng lóng, viết tắt, chửi thề và cách diễn đạt phụ thuộc ngữ cảnh. Một từ có vẻ tiêu cực chưa đủ để xác định mức độ vi phạm: cần xem câu nói đang bộc lộ cảm xúc chung hay công kích một đối tượng cụ thể. Dự án khảo sát khả năng dùng mô hình ngôn ngữ để **hỗ trợ sàng lọc nội dung cho người kiểm duyệt**, kèm lý do để thuận tiện kiểm tra lại quyết định.

Đầu vào là một bình luận; đầu ra gồm ba trường: **`label`** để phân loại, **`explanation`** để giải thích ngắn gọn bằng tiếng Việt và **`evidence`** để trích nguyên văn đoạn liên quan. Các nhãn được mô tả theo prompt của notebook:

| Nhãn               | Ý nghĩa trong thực nghiệm                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------- |
| **CLEAN**     | Không chứa nội dung quấy rối, ngôn từ thô tục hoặc công kích.                                     |
| **OFFENSIVE** | Có lời thô tục hoặc xúc phạm nhưng không nhắm vào một đối tượng cụ thể.                     |
| **HATE**      | Có nội dung lăng mạ, quấy rối hoặc công kích hướng vào một cá nhân hay nhóm người cụ thể. |

Đây là quy ước phân loại của thực nghiệm; khi áp dụng vào một nền tảng cụ thể, cần đối chiếu lại với chính sách kiểm duyệt của nền tảng đó.

Dữ liệu sử dụng là **[`vominhmanh/vihsd-explainable`](https://huggingface.co/datasets/vominhmanh/vihsd-explainable)**, bản mở rộng của ViHSD với **24.048 mẫu train**, **2.672 mẫu validation** và **6.680 mẫu test**. Notebook giữ nguyên các split có sẵn: train dùng để cập nhật adapter, validation dùng để theo dõi loss, test dùng để so sánh trước và sau fine-tune.

Theo dataset card, nhãn gốc kế thừa từ `htdung167/ViHSD`; phần **lời giải thích và bằng chứng được GPT-4o-mini trên Azure tạo**, sau đó được tác giả rà soát, tuyển chọn. Vì vậy, đây là dữ liệu có bổ sung giám sát từ mô hình giáo viên, không nên xem toàn bộ lời giải thích là chú giải thủ công độc lập. Dataset công bố giấy phép **CC-BY-4.0**. [Nguồn dữ liệu và xuất xứ chú giải](https://huggingface.co/datasets/vominhmanh/vihsd-explainable/blob/main/README.md).

Ví dụ minh họa cấu trúc đầu ra, **không phải một dự đoán được đo trong thực nghiệm**:

```json
{
  "explanation": "Câu thể hiện sự cảm ơn, không có nội dung công kích.",
  "evidence": [],
  "label": "CLEAN"
}
```

## Vì sao fine-tune và vì sao chọn Qwen3?

**Fine-tuning**, hay tinh chỉnh, là tiếp tục huấn luyện một mô hình đã có kiến thức nền trên dữ liệu của bài toán cụ thể. Có thể hiểu ngắn gọn là **dạy một mô hình đã biết ngôn ngữ cách thực hiện một công việc theo các ví dụ và quy ước mong muốn**.

Ở đây, mô hình gốc đã có thể đọc bình luận và trả lời theo hướng dẫn, nhưng việc tuân thủ ranh giới giữa CLEAN, OFFENSIVE và HATE vẫn cần được kiểm chứng. Fine-tune cung cấp các ví dụ nhất quán về bình luận, nhãn, lời giải thích và bằng chứng để mô hình học cách phản hồi phù hợp với bộ dữ liệu. Đây là lý do thiết kế phương pháp; kết quả F1 bên dưới cho thấy cải thiện ở **nhãn dự đoán**, còn chất lượng giải thích cần được đánh giá riêng.

Checkpoint được dùng là **`unsloth/Qwen3-4B-Instruct-2507`**, thuộc họ **Qwen3-4B-Instruct-2507**. Model card chính thức mô tả đây là mô hình causal language model khoảng **4 tỷ tham số**, hỗ trợ nhiều ngôn ngữ và chỉ dùng **non-thinking mode**. Các đặc điểm đó phù hợp với mục tiêu thử nghiệm: nhận hướng dẫn tiếng Việt và sinh một câu trả lời JSON gồm nhiều trường. Quy mô 4B cũng là lựa chọn thực dụng để kết hợp với adapter và lượng tử hóa. [Model card Qwen](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

## Phương pháp và công nghệ

Luồng thực nghiệm đi từ **đánh giá mô hình gốc → chuẩn bị hội thoại huấn luyện → gắn adapter → SFT → lưu adapter → đánh giá lại trên cùng tập test**.

**SFT — Supervised Fine-Tuning** là huấn luyện có giám sát từ các ví dụ đầu vào và đầu ra mong muốn. Mỗi mẫu được chuyển thành một hội thoại: phía `user` chứa định nghĩa nhãn và bình luận, phía `assistant` chứa JSON lấy từ `explanation`, `evidence` và `label` của dataset. Notebook áp dụng chat template **`qwen3-instruct`**, rồi dùng **TRL `SFTTrainer`** để học dự đoán token tiếp theo. [Tài liệu SFT Trainer](https://huggingface.co/docs/trl/sft_trainer).

**LoRA — Low-Rank Adaptation** giữ cố định trọng số nền và huấn luyện các ma trận bổ sung nhỏ, gọi là adapter. **QLoRA** kết hợp adapter LoRA với mô hình nền được lượng tử hóa **4-bit**, giúp giảm bộ nhớ lưu trọng số trong khi vẫn học được phần điều chỉnh cho tác vụ. Notebook hiện thực hướng này bằng `load_in_4bit=True`, `full_finetuning=False` và `get_peft_model()`. Không có cấu hình tường minh để khẳng định riêng loại lượng tử hóa hay double quantization; optimizer được chọn là **`adamw_8bit`**. [Bài báo QLoRA](https://arxiv.org/abs/2305.14314).

Các kỹ thuật bổ trợ gồm **FP16** cho tính toán, **gradient accumulation** để tích lũy gradient qua nhiều micro-batch, **gradient checkpointing** để giảm bộ nhớ activation và **sequence packing** để ghép mẫu vào các chuỗi dài tối đa 1.024 token. Log ghi nhận **33.030.144 tham số được huấn luyện**, tương đương khoảng **0,81%** tổng tham số được báo cáo. Sau packing, trainer báo **11.894 chuỗi train**, nên số optimizer steps không được tính trực tiếp bằng số bình luận gốc chia cho batch hiệu dụng.

| Nhóm cấu hình           | Giá trị trong notebook                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| Mô hình nền             | `unsloth/Qwen3-4B-Instruct-2507`, nạp 4-bit                                            |
| LoRA                       | `r=16`, `alpha=32`, `dropout=0`, `bias="none"`                                    |
| Vị trí gắn adapter      | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Độ dài chuỗi / packing | 1.024 token / bật                                                                        |
| Batch train / validation   | 2 / 2 trên mỗi GPU                                                                      |
| Gradient accumulation      | 2; batch hiệu dụng là 4 chuỗi đã packing trên một GPU                             |
| Epoch / optimizer steps    | 1 / 2.974 bước thực tế                                                                |
| Learning rate / scheduler  | `2e-4` / cosine, warmup 25 bước                                                       |
| Optimizer / precision      | `adamw_8bit` / FP16, `bf16=False`                                                     |
| Theo dõi / checkpoint     | Log mỗi 50 bước; eval và lưu mỗi 300 bước; giữ tối đa 2 checkpoint             |
| Hạ tầng chạy            | 1 **NVIDIA H200** trên Modal; job yêu cầu 4 CPU và 32 GiB RAM                   |

**Python, PyTorch và Hugging Face Transformers** cung cấp nền tảng mô hình và suy luận; **Datasets** tải và biến đổi dữ liệu; **Unsloth, PEFT/LoRA, TRL, bitsandbytes và Accelerate** hỗ trợ fine-tune và quản lý tính toán; **scikit-learn** tính F1. **Google Colab** điều phối notebook, **Modal** thực thi các job GPU và lưu dữ liệu trên Volume, **Google Drive** nhận bản sao adapter và kết quả, còn **Weights & Biases** theo dõi huấn luyện.

Một chi tiết ảnh hưởng đến cách đọc loss: dữ liệu được truyền vào trainer dưới dạng **chuỗi hội thoại đã định dạng**, và notebook không bật `train_on_responses_only` hay cấu hình chỉ tính loss trên phần assistant. Do đó, không nên diễn giải loss hiện tại như một phép đo riêng cho chất lượng nhãn hoặc phần JSON trả lời.

## Kết quả trước và sau fine-tune

Cả hai lần đánh giá đều dùng **6.680 bình luận test**, cùng prompt định nghĩa nhãn và schema JSON, batch size **32**, mô hình nạp **4-bit**, `max_new_tokens=512` và `temperature=0.7`. Code lấy trường `label` từ JSON, chuẩn hóa bằng `strip().upper()`, rồi gọi `f1_score(..., average="macro")`.

| Chỉ số được ghi nhận                                        | Mô hình gốc, chưa fine-tune theo tác vụ |                                            Sau SFT + QLoRA |
| ----------------------------------------------------------------- | --------------------------------------------: | ---------------------------------------------------------: |
| Số mẫu test                                                     |                                         6.680 |                                                      6.680 |
| **Macro F1 theo code hiện tại**                           |                              **0,3721** |                                           **0,6207** |
| Mức tăng tuyệt đối                                           |                                            — | **+0,2486**, tức **24,86 điểm phần trăm** |
| Số đầu ra bị gán `ERROR` do lỗi đọc JSON/trường nhãn |                      Không được in riêng |                                                **0** |

**Macro F1** lấy trung bình F1 của từng lớp, trong đó F1 cân bằng giữa precision và recall. Với tập nhãn cố định, cách tính này giúp mỗi lớp có trọng số ngang nhau thay vì để lớp nhiều mẫu chi phối điểm tổng.

**Lưu ý về điểm đang báo cáo:** notebook không truyền danh sách `labels` cố định cho `f1_score`. Nếu xuất hiện dự đoán `ERROR` hoặc một nhãn ngoài quy định, scikit-learn có thể đưa nhãn đó vào tập lớp để tính trung bình. Kết quả trên được giữ nguyên theo output đã lưu; chưa có đủ nhãn dự đoán baseline trong thư mục để xác nhận lại cùng một mẫu số ba lớp. Vì vậy, cần đọc mức tăng như **so sánh theo phép tính hiện tại**, và tính lại với `labels=["CLEAN", "OFFENSIVE", "HATE"]`, kèm tỷ lệ đầu ra lỗi, khi cần một báo cáo ba lớp thống nhất. [Quy ước `f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Run huấn luyện có tên **`run-fdfeaea1d367`**, tương ứng W&B ID **`n9qhscs3`**. Notebook ghi nhận **training loss trung bình 0,2274888041** sau **2.974 bước**. Đây là trung bình của quá trình train, khác với giá trị train loss tại một thời điểm trên biểu đồ.

![Ảnh chụp W&B: train loss, learning rate, gradient norm, global step và epoch](screenshots/User%20attachment2.png)

*Ảnh chụp màn hình W&B của run `run-fdfeaea1d367`, gồm các chỉ số huấn luyện.*

Train loss giảm mạnh ở giai đoạn đầu rồi dao động ở mức thấp; learning rate tăng trong warmup và giảm dần về gần 0 theo cosine scheduler. Diễn biến này phù hợp với việc mô hình nhanh chóng thích nghi với cấu trúc dữ liệu, nhưng riêng train loss chưa cho biết mô hình phân biệt tốt từng nhãn đến đâu.

![Ảnh chụp W&B: validation loss, runtime, samples/second và steps/second](screenshots/User%20attachment.png)

*Ảnh chụp màn hình W&B của cùng run, gồm validation loss và các chỉ số thực thi đánh giá. Xem biểu đồ tương tác tại [run gốc](https://forge.coreweave.com/wandb/giangnguyeen1910-luluchi19/qwen3-vihsd/runs/n9qhscs3).*

Validation loss trên ảnh giảm từ khoảng **0,225 xuống 0,204** và chậm dần về cuối. Trong khoảng quan sát này chưa thấy validation loss tăng trở lại rõ rệt; dữ liệu đó chưa đủ để kết luận mô hình không overfit. Các biểu đồ runtime và samples/second trong ảnh W&B là thống kê vòng validation của trainer, không phải benchmark suy luận phân loại đầu cuối trên bình luận.

**Phạm vi kết luận:** thực nghiệm cho thấy điểm phân loại được báo cáo cải thiện sau fine-tune. `ERROR=0` chỉ có nghĩa parser đọc được JSON và trường nhãn ở lần đánh giá sau train, chưa chứng minh toàn bộ schema hợp lệ hay bằng chứng trích dẫn luôn đúng. Notebook chưa báo cáo accuracy, precision/recall/F1 từng lớp, confusion matrix hoặc chất lượng `explanation`/`evidence`. Code cũng chưa cố định đầy đủ cấu hình sinh như `do_sample`, `top_p`, `top_k`, nên kết quả không được xem là bảo đảm tái lập tuyệt đối giữa các môi trường.

Các bước đánh giá tiếp theo hợp lý là thống nhất phép đo ba lớp, kiểm tra lỗi nhầm OFFENSIVE–HATE, rà soát bằng chứng có nằm nguyên văn trong đầu vào và thử trên bình luận ngoài ViHSD. Dự án hiện là **thực nghiệm huấn luyện và đánh giá trong notebook**, chưa có đánh giá triển khai kiểm duyệt thực tế.

## Tài liệu tham khảo

Nguồn kỹ thuật chính: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [ViHSD Explainable](https://huggingface.co/datasets/vominhmanh/vihsd-explainable), [QLoRA](https://arxiv.org/abs/2305.14314), [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) và [Unsloth notebooks](https://github.com/unslothai/notebooks).
