-- 관리자 계정으로 실행하기 전에 비밀번호를 직접 수정하세요.
CREATE DATABASE IF NOT EXISTS look_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'look_user'@'localhost' IDENTIFIED BY 'CHANGE_ME';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX, REFERENCES ON look_db.* TO 'look_user'@'localhost';
-- 기존 계정이 있다면 새 계정 대신 기존 계정을 .env에서 사용해도 됩니다.
-- 테이블은 웹서버 최초 실행 때 자동 생성됩니다.
