-- Примеры SQL-запросов для ручной проверки метрик.

-- 1. Общие метрики
SELECT
    COUNT(*) AS total_orders,
    ROUND(SUM(total_price), 2) AS gmv,
    ROUND(AVG(total_price), 2) AS aov,
    ROUND(AVG(review_score), 3) AS avg_review_score,
    ROUND(AVG(is_bad_review), 3) AS bad_review_rate,
    ROUND(AVG(is_delayed), 3) AS delay_rate,
    ROUND(AVG(delivery_time_days), 2) AS avg_delivery_time_days,
    ROUND(AVG(is_cancelled), 3) AS cancelled_rate
FROM orders_mart;

-- 2. Метрики по месяцам
SELECT
    order_year_month,
    COUNT(*) AS orders,
    ROUND(SUM(total_price), 2) AS gmv,
    ROUND(AVG(total_price), 2) AS aov,
    ROUND(AVG(review_score), 3) AS avg_review_score,
    ROUND(AVG(is_bad_review), 3) AS bad_review_rate,
    ROUND(AVG(is_delayed), 3) AS delay_rate
FROM orders_mart
GROUP BY order_year_month
ORDER BY order_year_month;

-- 3. Категории с худшей долей плохих отзывов
SELECT
    product_category_name,
    COUNT(*) AS orders,
    ROUND(SUM(total_price), 2) AS gmv,
    ROUND(AVG(review_score), 3) AS avg_review_score,
    ROUND(AVG(is_bad_review), 3) AS bad_review_rate,
    ROUND(AVG(is_delayed), 3) AS delay_rate
FROM orders_mart
WHERE product_category_name IS NOT NULL
GROUP BY product_category_name
HAVING COUNT(*) >= 100
ORDER BY bad_review_rate DESC, orders DESC
LIMIT 20;

-- 4. Регионы покупателей с высокой долей задержек
SELECT
    customer_state,
    COUNT(*) AS orders,
    ROUND(AVG(is_delayed), 3) AS delay_rate,
    ROUND(AVG(delivery_time_days), 2) AS avg_delivery_time_days,
    ROUND(AVG(review_score), 3) AS avg_review_score
FROM orders_mart
WHERE customer_state IS NOT NULL
GROUP BY customer_state
HAVING COUNT(*) >= 100
ORDER BY delay_rate DESC, orders DESC;
