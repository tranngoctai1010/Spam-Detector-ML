🔹 General Logging Style Guide
Keep it structured → [Component] - [Action/Issue] - [Details]
Use active voice → "Failed to load" instead of "Loading was failed"
Be specific → "Invalid input format" instead of "Something went wrong"
Provide solutions if possible

🔹 Logging Messages by Level
✅ INFO (General process updates)
[DataLoader] - Successfully loaded dataset from %s
[Preprocessing] - Text tokenization completed. %d samples processed.
[Training] - Model training started with batch size %d and learning rate %.5f
[Evaluation] - Accuracy achieved: %.2f%% on validation set
[Inference] - Prediction completed in %.3f seconds.
[Pipeline] - Process completed successfully with no errors detected.

⚠️ WARNING (Unexpected behavior, but execution continues)
[Preprocessing] - Found %d missing values, filling with default.
[Model] - Using CPU instead of GPU due to compatibility issues.
[Training] - Convergence warning: Loss has not decreased for %d epochs.
[Evaluation] - Low performance detected: Accuracy dropped below expected threshold.
[System] - High memory usage detected, consider optimizing model size.

❌ ERROR (Something failed, but the system continues)
[DataLoader] - Failed to read file: %s. Please check the file path.
[Preprocessing] - Tokenization error: %s. Invalid input detected.
[Training] - Model checkpoint not found: %s. Training cannot resume.
[Inference] - Unable to process request due to missing input parameters.
[API] - Response failed with status code %d: %s

🚨 CRITICAL (Fatal errors, system failure, must stop execution)
[System] - GPU is unavailable, terminating process.
[Training] - Detected NaN values in gradients, stopping training to prevent corruption.
[Security] - Unauthorized access attempt detected, shutting down service.
[Model] - Weight corruption detected, aborting model initialization.
[Database] - Connection lost. Unable to recover.





🔹 Hướng dẫn chung về ghi log (General Logging Style Guide)
Giữ cấu trúc rõ ràng → [Thành phần] - [Hành động/Vấn đề] - [Chi tiết]
Sử dụng câu chủ động → "Không thể tải" thay vì "Việc tải đã bị thất bại"
Cụ thể, rõ ràng → "Định dạng đầu vào không hợp lệ" thay vì "Có lỗi xảy ra"
Cung cấp giải pháp nếu có thể
🔹 Các mức ghi log (Logging Messages by Level)
✅ INFO (Cập nhật quá trình chung)

[DataLoader] - Đã tải thành công tập dữ liệu từ %s
[Preprocessing] - Hoàn tất tokenization văn bản. Đã xử lý %d mẫu.
[Training] - Bắt đầu huấn luyện mô hình với batch size %d và learning rate %.5f
[Evaluation] - Đạt độ chính xác: %.2f%% trên tập validation.
[Inference] - Hoàn thành dự đoán trong %.3f giây.
[Pipeline] - Quá trình hoàn tất mà không có lỗi.
⚠️ WARNING (Hành vi bất thường, nhưng hệ thống vẫn tiếp tục chạy)

[Preprocessing] - Tìm thấy %d giá trị bị thiếu, đang điền giá trị mặc định.
[Model] - Đang sử dụng CPU thay vì GPU do vấn đề tương thích.
[Training] - Cảnh báo hội tụ: Loss không giảm trong %d epochs.
[Evaluation] - Hiệu suất thấp: Độ chính xác giảm xuống dưới ngưỡng mong đợi.
[System] - Bộ nhớ sử dụng cao, hãy cân nhắc tối ưu kích thước mô hình.
❌ ERROR (Lỗi nghiêm trọng, nhưng hệ thống vẫn tiếp tục chạy)

[DataLoader] - Không thể đọc tệp: %s. Vui lòng kiểm tra đường dẫn.
[Preprocessing] - Lỗi tokenization: %s. Đầu vào không hợp lệ.
[Training] - Không tìm thấy checkpoint mô hình: %s. Không thể tiếp tục huấn luyện.
[Inference] - Không thể xử lý yêu cầu do thiếu tham số đầu vào.
[API] - Phản hồi thất bại với mã trạng thái %d: %s
🚨 CRITICAL (Lỗi nghiêm trọng, hệ thống phải dừng ngay lập tức)

[System] - GPU không khả dụng, dừng quá trình.
[Training] - Phát hiện giá trị NaN trong gradients, dừng huấn luyện để tránh lỗi.
[Security] - Phát hiện truy cập trái phép, đang tắt dịch vụ.
[Model] - Phát hiện trọng số bị lỗi, hủy khởi tạo mô hình.
[Database] - Mất kết nối với cơ sở dữ liệu. Không thể khôi phục.
