-- Витрина заказов для учебного аналитического проекта.
-- Уровень данных: одна строка = один заказ.
-- SQL написан под SQLite.

DROP TABLE IF EXISTS order_items_agg;
DROP TABLE IF EXISTS payments_agg;
DROP TABLE IF EXISTS reviews_agg;
DROP TABLE IF EXISTS orders_mart;

CREATE TABLE order_items_agg AS
WITH item_with_category AS (
    SELECT
        oi.order_id,
        oi.order_item_id,
        oi.product_id,
        oi.seller_id,
        CAST(oi.price AS REAL) AS price,
        CAST(oi.freight_value AS REAL) AS freight_value,
        CASE
            WHEN p.product_category_name IS NULL OR p.product_category_name = ''
                THEN 'missing_product_category'
            WHEN t.product_category_name_english IS NULL
                THEN 'missing_translation'
            ELSE t.product_category_name_english
        END AS product_category_name,
        s.seller_state
    FROM raw_order_items AS oi
    LEFT JOIN raw_products AS p
        ON oi.product_id = p.product_id
    LEFT JOIN raw_category_translation AS t
        ON p.product_category_name = t.product_category_name
    LEFT JOIN raw_sellers AS s
        ON oi.seller_id = s.seller_id
),
order_totals AS (
    SELECT
        order_id,
        COUNT(*) AS order_item_count,
        COUNT(DISTINCT product_id) AS distinct_products,
        COUNT(DISTINCT seller_id) AS distinct_sellers,
        COUNT(DISTINCT product_category_name) AS distinct_categories,
        SUM(price) AS total_price,
        SUM(freight_value) AS total_freight
    FROM item_with_category
    GROUP BY order_id
),
main_category AS (
    SELECT order_id, product_category_name
    FROM (
        SELECT
            order_id,
            product_category_name,
            ROW_NUMBER() OVER (
                PARTITION BY order_id
                ORDER BY SUM(price) DESC, product_category_name
            ) AS rn
        FROM item_with_category
        GROUP BY order_id, product_category_name
    )
    WHERE rn = 1
),
main_seller AS (
    SELECT order_id, seller_state
    FROM (
        SELECT
            order_id,
            seller_state,
            ROW_NUMBER() OVER (
                PARTITION BY order_id
                ORDER BY SUM(price) DESC, seller_state
            ) AS rn
        FROM item_with_category
        GROUP BY order_id, seller_state
    )
    WHERE rn = 1
)
SELECT
    ot.order_id,
    ot.order_item_count,
    ot.distinct_products,
    ot.distinct_sellers,
    ot.distinct_categories,
    ot.total_price,
    ot.total_freight,
    mc.product_category_name,
    ms.seller_state,
    CASE WHEN ot.distinct_categories > 1 THEN 1 ELSE 0 END AS is_mixed_category
FROM order_totals AS ot
LEFT JOIN main_category AS mc
    ON ot.order_id = mc.order_id
LEFT JOIN main_seller AS ms
    ON ot.order_id = ms.order_id;

CREATE TABLE payments_agg AS
WITH payment_totals AS (
    SELECT
        order_id,
        SUM(CAST(payment_value AS REAL)) AS payment_value,
        COUNT(*) AS payment_count,
        MAX(CAST(payment_installments AS INTEGER)) AS max_installments
    FROM raw_order_payments
    GROUP BY order_id
),
main_payment AS (
    SELECT order_id, payment_type
    FROM (
        SELECT
            order_id,
            payment_type,
            ROW_NUMBER() OVER (
                PARTITION BY order_id
                ORDER BY SUM(CAST(payment_value AS REAL)) DESC, payment_type
            ) AS rn
        FROM raw_order_payments
        GROUP BY order_id, payment_type
    )
    WHERE rn = 1
)
SELECT
    pt.order_id,
    pt.payment_value,
    pt.payment_count,
    pt.max_installments,
    mp.payment_type AS main_payment_type
FROM payment_totals AS pt
LEFT JOIN main_payment AS mp
    ON pt.order_id = mp.order_id;

CREATE TABLE reviews_agg AS
SELECT
    order_id,
    AVG(CAST(review_score AS REAL)) AS review_score,
    MIN(CAST(review_score AS INTEGER)) AS min_review_score,
    COUNT(*) AS review_count,
    MAX(CASE WHEN CAST(review_score AS INTEGER) <= 2 THEN 1 ELSE 0 END) AS is_bad_review
FROM raw_order_reviews
GROUP BY order_id;

CREATE TABLE orders_mart AS
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    substr(o.order_purchase_timestamp, 1, 7) AS order_year_month,

    c.customer_city,
    c.customer_state,
    ia.seller_state,
    CASE
        WHEN ia.order_id IS NULL THEN 'no_order_items'
        ELSE ia.product_category_name
    END AS product_category_name,
    COALESCE(ia.is_mixed_category, 0) AS is_mixed_category,

    COALESCE(ia.order_item_count, 0) AS order_item_count,
    COALESCE(ia.distinct_products, 0) AS distinct_products,
    COALESCE(ia.distinct_sellers, 0) AS distinct_sellers,
    COALESCE(ia.distinct_categories, 0) AS distinct_categories,
    COALESCE(ia.total_price, 0.0) AS total_price,
    COALESCE(ia.total_freight, 0.0) AS total_freight,
    COALESCE(pa.payment_value, 0.0) AS payment_value,
    COALESCE(pa.payment_count, 0) AS payment_count,
    COALESCE(pa.max_installments, 0) AS max_installments,
    COALESCE(pa.main_payment_type, 'unknown') AS main_payment_type,

    ra.review_score,
    ra.min_review_score,
    COALESCE(ra.review_count, 0) AS review_count,
    CASE WHEN ra.review_count > 0 THEN 1 ELSE 0 END AS has_review,
    ra.is_bad_review,

    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL AND o.order_delivered_customer_date != '' THEN 1
        ELSE 0
    END AS is_delivered,

    CASE
        WHEN o.order_status = 'canceled' THEN 1
        ELSE 0
    END AS is_cancelled,

    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL AND o.order_delivered_customer_date != ''
        THEN julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)
        ELSE NULL
    END AS delivery_time_days,

    CASE
        WHEN o.order_estimated_delivery_date IS NOT NULL AND o.order_estimated_delivery_date != ''
        THEN julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp)
        ELSE NULL
    END AS estimated_delivery_time_days,

    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL AND o.order_delivered_customer_date != ''
             AND o.order_estimated_delivery_date IS NOT NULL AND o.order_estimated_delivery_date != ''
        THEN julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date)
        ELSE NULL
    END AS delay_days,

    CASE
        WHEN o.order_delivered_customer_date IS NULL OR o.order_delivered_customer_date = ''
             OR o.order_estimated_delivery_date IS NULL OR o.order_estimated_delivery_date = ''
        THEN NULL
        WHEN julianday(o.order_delivered_customer_date) > julianday(o.order_estimated_delivery_date)
        THEN 1
        ELSE 0
    END AS is_delayed,

    CASE
        WHEN COALESCE(ia.total_price, 0.0) > 0
        THEN COALESCE(ia.total_freight, 0.0) / COALESCE(ia.total_price, 0.0)
        ELSE NULL
    END AS freight_share

FROM raw_orders AS o
LEFT JOIN raw_customers AS c
    ON o.customer_id = c.customer_id
LEFT JOIN order_items_agg AS ia
    ON o.order_id = ia.order_id
LEFT JOIN payments_agg AS pa
    ON o.order_id = pa.order_id
LEFT JOIN reviews_agg AS ra
    ON o.order_id = ra.order_id;

CREATE INDEX IF NOT EXISTS idx_orders_mart_order_id ON orders_mart(order_id);
CREATE INDEX IF NOT EXISTS idx_orders_mart_month ON orders_mart(order_year_month);
CREATE INDEX IF NOT EXISTS idx_orders_mart_category ON orders_mart(product_category_name);
CREATE INDEX IF NOT EXISTS idx_orders_mart_customer_state ON orders_mart(customer_state);
