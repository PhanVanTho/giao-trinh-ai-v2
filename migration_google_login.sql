-- Migration: Thêm cột hỗ trợ Google Sign-In cho bảng nguoi_dung
-- Chạy lệnh này trong phpMyAdmin hoặc MySQL CLI

ALTER TABLE `nguoi_dung` 
    ADD COLUMN `google_id` VARCHAR(255) NULL DEFAULT NULL AFTER `email`,
    ADD COLUMN `ho_ten` VARCHAR(255) NULL DEFAULT NULL AFTER `google_id`,
    ADD COLUMN `anh_dai_dien` VARCHAR(500) NULL DEFAULT NULL AFTER `ho_ten`,
    ADD UNIQUE INDEX `uq_google_id` (`google_id`);

-- Cho phép mat_khau_hash NULL (cho user đăng nhập bằng Google không cần password)
ALTER TABLE `nguoi_dung` 
    MODIFY COLUMN `mat_khau_hash` VARCHAR(255) NULL DEFAULT NULL;
