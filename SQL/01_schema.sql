-- ==========================================================
-- Book Recommendation System - Database Schema
-- Compatible with MySQL, PostgreSQL, SQLite
-- ==========================================================

-- Disable foreign key checks for bulk import
-- For MySQL:
-- SET FOREIGN_KEY_CHECKS = 0;
-- SET SQL_MODE = 'NO_AUTO_VALUE_ON_ZERO';
-- SET autocommit = 0;

-- For PostgreSQL:
-- SET session_replication_role = 'replica';

-- 1. Table: users
DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    user_id INT PRIMARY KEY,
    location VARCHAR(255),
    age INT
);

-- 2. Table: books
CREATE TABLE books (
    isbn VARCHAR(20) PRIMARY KEY,
    book_title VARCHAR(500) NOT NULL,
    book_author VARCHAR(255),
    year_of_publication INT,
    publisher VARCHAR(255),
    image_url_s VARCHAR(500),
    image_url_m VARCHAR(500),
    image_url_l VARCHAR(500)
);

-- 3. Table: ratings
CREATE TABLE ratings (
    user_id INT NOT NULL,
    isbn VARCHAR(20) NOT NULL,
    book_rating TINYINT NOT NULL,
    PRIMARY KEY (user_id, isbn)
);

-- Recommended Indexes for RecSys Performance
CREATE INDEX idx_ratings_user ON ratings (user_id);
CREATE INDEX idx_ratings_isbn ON ratings (isbn);
CREATE INDEX idx_ratings_score ON ratings (book_rating);
CREATE INDEX idx_books_author ON books (book_author);
CREATE INDEX idx_books_year ON books (year_of_publication);
CREATE INDEX idx_users_age ON users (age);
