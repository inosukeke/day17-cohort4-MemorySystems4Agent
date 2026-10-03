# Phân tích kết quả benchmark

Số liệu dưới đây lấy từ một lần chạy `python src/benchmark.py` ở chế độ offline (không cần API key).

## Standard Benchmark (`data/conversations.json`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------:|-------------------------:|----------------------:|-------------------:|------------------------:|-------------:|
| Baseline | 1221 | 12666 | 0.00 | 0.30 | 0 | 0 |
| Advanced | 2733 | 24122 | 1.00 | 1.00 | 315 | 0 |

## Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------:|-------------------------:|----------------------:|-------------------:|------------------------:|-------------:|
| Baseline | 346 | 22649 | 0.00 | 0.30 | 0 | 0 |
| Advanced | 522 | 14384 | 1.00 | 1.00 | 282 | 11 |

## Vì sao Advanced có recall tốt hơn Baseline

Recall question luôn được hỏi ở **thread mới**. Baseline chỉ giữ state trong `SessionState` theo `thread_id`, nên sang thread mới nó không còn bất kỳ fact nào — recall = 0 trên cả hai bộ dữ liệu. Advanced ghi fact ổn định (tên, nơi ở, nghề nghiệp, đồ uống, món ăn, thú cưng, sở thích, style trả lời) vào `User.md` theo `user_id`, độc lập với `thread_id`, nên nó đọc lại được bất kể thread nào — recall = 1.0.

## Vì sao Advanced có thể tốn hơn ở hội thoại ngắn

Ở Standard Benchmark, Advanced tốn gấp ~2.2x token (2733 vs 1221) và gấp ~1.9x prompt tokens (24122 vs 12666) so với Baseline. Lý do: mỗi lượt Advanced đều phải đọc `User.md` + summary + message gần nhất để trả lời, trong khi hội thoại 10 lượt/conversation còn quá ngắn để compact memory phát huy tác dụng (0 compaction ở Standard Benchmark — ngưỡng token chưa bị vượt). Nghĩa là ở quy mô nhỏ, chi phí duy trì persistent memory lớn hơn lợi ích nó mang lại về mặt token.

## Vì sao compact giúp Advanced có lợi thế ở hội thoại dài

Ở Long-Context Stress Benchmark (1 hội thoại 16 lượt rất dài), Advanced compact 11 lần và kết quả là **prompt tokens processed thấp hơn Baseline** (14384 vs 22649) dù vẫn đạt recall = 1.0. Baseline phải mang theo toàn bộ lịch sử message tăng dần qua từng lượt (không có cơ chế nén), nên chi phí ngữ cảnh tăng tuyến tính theo độ dài hội thoại. Advanced nén các message cũ thành summary mỗi khi vượt `compact_threshold_tokens`, chỉ giữ lại `compact_keep_messages` message gần nhất + summary + `User.md`, nên ngữ cảnh không phình vô hạn. Điều này đúng như thiết kế: **compact tối ưu chủ yếu ở `prompt tokens processed`**, không phải ở `agent tokens only`.

## File memory tăng trưởng và rủi ro

`User.md` của Advanced tăng từ 0 lên 315 bytes (Standard) / 282 bytes (Stress) sau một bộ hội thoại — nhỏ vì chỉ lưu fact có cấu trúc (`- **key**: value`), không lưu nguyên văn hội thoại. Rủi ro đi kèm:

- **Phình file theo thời gian**: nếu chạy hàng trăm phiên, số fact có thể tăng và file khó đọc nếu không có cơ chế dọn dẹp (memory decay).
- **Lưu sai fact khi người dùng chỉ hỏi chứ không cung cấp thông tin**: đây là lỗi thực tế gặp phải khi làm lab này — câu hỏi "Tên mình là gì?" ban đầu bị `extract_profile_updates()` hiểu nhầm thành phát biểu và ghi đè `name` bằng rác. Đã sửa bằng cách bỏ qua toàn bộ câu có dấu `?` khi trích fact (coi câu hỏi là *cầu xin thông tin*, không phải *cung cấp thông tin*).
- **Correction/conflict**: dữ liệu benchmark cố tình đổi nơi ở (Đà Nẵng ↔ Huế ↔ Đà Nẵng) và nghề nghiệp (backend engineer → MLOps engineer), kèm nhiễu (Hà Nội chỉ là nơi họp, "product manager" chỉ là câu đùa). `extract_profile_updates()` phát hiện các cụm phủ định ("không còn", "chứ không", "đừng nói ... nữa") và ngữ cảnh đùa ("đùa", "giỡn") để không ghi fact cũ/sai đè lên fact đúng, và `upsert_fact()` luôn thay thế toàn bộ dòng fact cũ thay vì cộng dồn — nên không bao giờ tồn tại đồng thời hai giá trị mâu thuẫn cho cùng một key.

## Bonus đã áp dụng

- **Entity extraction có cấu trúc**: fact được tách thành các field riêng (`name`, `location`, `profession`, `style`, `favorite_drink`, `favorite_food`, `pet`, `interests`) thay vì một blob văn bản.
- **Conflict handling**: negation/joke detection + `upsert_fact()` ghi đè đúng dòng, đảm bảo correction mới nhất luôn thắng.
