import logging
import logging.config
import yaml

def setup_logging(config_path="configs/logging_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

setup_logging()

def get_logger(name):
    return logging.getLogger(name)






# Sử dụng logging trong các module
from logging_config import get_logger

logger = get_logger("app")

def run_pipeline():
    logger.info("Pipeline bắt đầu chạy...")
    try:
        # Code chính ở đây
        logger.debug("Đang xử lý batch dữ liệu đầu tiên")
    except Exception as e:
        logger.error(f"Lỗi xảy ra: {str(e)}", exc_info=True)
        raise e

def critical_failure():
    logger.critical("Lỗi nghiêm trọng! Dừng toàn bộ hệ thống.")

run_pipeline()
critical_failure()






# ✅ Kết quả
# logs/info.log sẽ chứa các log từ cấp INFO trở lên
# logs/error.log sẽ chứa các log từ cấp ERROR trở lên
# logs/critical.log sẽ chứa chỉ các log CRITICAL
# Console sẽ hiển thị toàn bộ log từ DEBUG trở lên
# Cách này giúp bạn dễ dàng phân tích log theo cấp độ, phù hợp cho dự án lớn. 🚀




# Nếu muốn mở rộng, bạn có thể:
# ✔ Thêm handler gửi ERROR/CRITICAL vào email (dùng SMTPHandler)
# ✔ Ghi log ra JSON thay vì text (dùng logging.handlers.RotatingFileHandler)
# ✔ Đồng bộ log lên hệ thống giám sát như ELK, Grafana