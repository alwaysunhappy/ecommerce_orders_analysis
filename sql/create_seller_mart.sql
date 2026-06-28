-- Витрины для анализа продавцов.
-- order_seller: одна строка = один продавец внутри одного заказа (grain = order_id x seller_id).
-- seller_mart:  одна строка = один продавец (агрегат по всем его заказам).
-- Требует уже созданную таблицу orders_mart (запускать после create_orders_mart.sql).
-- SQL написан под SQLite.

DROP TABLE IF EXISTS order_seller;
DROP TABLE IF EXISTS seller_mart;

CREATE TABLE order_seller AS
WITH item_base AS (
    SELECT
        oi.order_id,
        oi.seller_id,
        CAST(oi.price AS REAL) AS price,
        CAST(oi.freight_value AS REAL) AS freight_value,
        oi.shipping_limit_date,
        CASE
            WHEN p.product_category_name IS NULL OR p.product_category_name = ''
                THEN 'missing_product_category'
            WHEN t.product_category_name_english IS NULL
                THEN 'missing_translation'
            ELSE t.product_category_name_english
        END AS product_category_name
    FROM raw_order_items AS oi
    LEFT JOIN raw_products AS p
        ON oi.product_id = p.product_id
    LEFT JOIN raw_category_translation AS t
        ON p.product_category_name = t.product_category_name
),
seller_totals AS (
    SELECT
        order_id,
        seller_id,
        COUNT(*) AS seller_item_count,
        SUM(price) AS seller_price,
        SUM(freight_value) AS seller_freight,
        MAX(shipping_limit_date) AS seller_shipping_limit_date
    FROM item_base
    GROUP BY order_id, seller_id
),
seller_main_category AS (
    SELECT order_id, seller_id, product_category_name
    FROM (
        SELECT
            order_id,
            seller_id,
            product_category_name,
            ROW_NUMBER() OVER (
                PARTITION BY order_id, seller_id
                ORDER BY SUM(price) DESC, product_category_name
            ) AS rn
        FROM item_base
        GROUP BY order_id, seller_id, product_category_name
    )
    WHERE rn = 1
)
SELECT
    st.order_id,
    st.seller_id,
    s.seller_state,
    s.seller_city,
    smc.product_category_name,
    st.seller_item_count,
    st.seller_price,
    st.seller_freight,
    om.distinct_sellers,
    CASE WHEN om.distinct_sellers = 1 THEN 1 ELSE 0 END AS is_single_seller_order,
    om.order_year_month,
    om.order_purchase_timestamp,
    om.customer_state,
    om.is_delivered,
    om.is_cancelled,
    om.is_delayed,
    om.delivery_time_days,
    om.handover_time_days,
    om.transit_time_days,
    om.review_score,
    om.is_bad_review,
    om.has_review,
    CASE
        WHEN om.order_delivered_carrier_date IS NULL OR om.order_delivered_carrier_date = ''
             OR st.seller_shipping_limit_date IS NULL OR st.seller_shipping_limit_date = ''
        THEN NULL
        WHEN julianday(om.order_delivered_carrier_date) > julianday(st.seller_shipping_limit_date)
        THEN 1
        ELSE 0
    END AS is_late_handover
FROM seller_totals AS st
LEFT JOIN raw_sellers AS s
    ON st.seller_id = s.seller_id
LEFT JOIN seller_main_category AS smc
    ON st.order_id = smc.order_id AND st.seller_id = smc.seller_id
LEFT JOIN orders_mart AS om
    ON st.order_id = om.order_id;

CREATE TABLE seller_mart AS
WITH seller_top_category AS (
    SELECT seller_id, product_category_name
    FROM (
        SELECT
            seller_id,
            product_category_name,
            ROW_NUMBER() OVER (
                PARTITION BY seller_id
                ORDER BY SUM(seller_price) DESC, product_category_name
            ) AS rn
        FROM order_seller
        GROUP BY seller_id, product_category_name
    )
    WHERE rn = 1
)
SELECT
    os.seller_id,
    MAX(os.seller_state) AS seller_state,
    MAX(os.seller_city) AS seller_city,
    MAX(stc.product_category_name) AS top_category,
    COUNT(DISTINCT os.order_id) AS orders,
    SUM(os.seller_item_count) AS items,
    COUNT(DISTINCT os.product_category_name) AS distinct_categories,
    SUM(os.seller_price) AS gmv,
    SUM(os.seller_freight) AS freight,
    SUM(os.is_single_seller_order) AS single_seller_orders,
    AVG(os.is_delivered) AS delivered_rate,
    AVG(os.is_delayed) AS delay_rate,
    AVG(os.is_bad_review) AS bad_review_rate,
    AVG(os.review_score) AS avg_review_score,
    AVG(os.has_review) AS review_coverage,
    AVG(os.delivery_time_days) AS avg_delivery_time_days,
    AVG(os.handover_time_days) AS avg_handover_time_days,
    AVG(os.transit_time_days) AS avg_transit_time_days,
    AVG(os.is_late_handover) AS late_handover_rate
FROM order_seller AS os
LEFT JOIN seller_top_category AS stc
    ON os.seller_id = stc.seller_id
GROUP BY os.seller_id;

CREATE INDEX IF NOT EXISTS idx_order_seller_order_id ON order_seller(order_id);
CREATE INDEX IF NOT EXISTS idx_order_seller_seller_id ON order_seller(seller_id);
CREATE INDEX IF NOT EXISTS idx_order_seller_category ON order_seller(product_category_name);
CREATE INDEX IF NOT EXISTS idx_seller_mart_seller_id ON seller_mart(seller_id);
