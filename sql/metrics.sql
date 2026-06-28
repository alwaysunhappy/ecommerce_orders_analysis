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

-- 5. Продавцы с худшими метриками клиентского опыта
SELECT
    seller_id,
    seller_state,
    top_category,
    orders,
    ROUND(delay_rate, 3) AS delay_rate,
    ROUND(late_handover_rate, 3) AS late_handover_rate,
    ROUND(bad_review_rate, 3) AS bad_review_rate,
    ROUND(avg_review_score, 3) AS avg_review_score
FROM seller_mart
WHERE orders >= 30
ORDER BY bad_review_rate DESC, orders DESC
LIMIT 20;

-- 6. Разброс продавцов внутри категории (категория vs отдельные продавцы)
SELECT
    product_category_name,
    seller_id,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(AVG(is_bad_review), 3) AS bad_review_rate,
    ROUND(AVG(is_delayed), 3) AS delay_rate,
    ROUND(AVG(is_late_handover), 3) AS late_handover_rate
FROM order_seller
WHERE is_single_seller_order = 1
GROUP BY product_category_name, seller_id
HAVING COUNT(DISTINCT order_id) >= 30
ORDER BY product_category_name, bad_review_rate DESC;

-- 7. Декомпозиция времени доставки по этапам и статусу задержки
SELECT
    is_delayed,
    COUNT(*) AS orders,
    ROUND(AVG(handover_time_days), 2) AS avg_handover_days,
    ROUND(AVG(transit_time_days), 2) AS avg_transit_days,
    ROUND(AVG(delivery_time_days), 2) AS avg_total_days
FROM orders_mart
WHERE is_delivered = 1
GROUP BY is_delayed;
