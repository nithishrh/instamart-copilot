-- Q1: WHERE then HAVING
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS high_value_spend,
    COUNT(*) AS high_value_order_count
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY high_value_spend DESC;

-- Q2: Three-table join
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue,
    COUNT(*) AS completed_orders
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: Scalar subquery
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN
SELECT
    c.customer_id,
    c.city,
    c.signup_date,
    c.age,
    c.gender
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Common Table Expression (CTE)
WITH customer_spend AS (
    SELECT
        customer_id,
        ROUND(SUM(order_amount), 2) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    cs.total_spend,
    ROUND(a.avg_total_spend, 2) AS avg_total_spend
FROM customer_spend cs
CROSS JOIN avg_spend a
WHERE cs.total_spend > a.avg_total_spend
ORDER BY cs.total_spend DESC;

-- Q6: Ranking window functions
WITH customer_city_spend AS (
    SELECT
        o.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_city_spend
ORDER BY city, row_num;

-- Q7: Running total
WITH daily_rev AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    revenue,
    ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily_rev
ORDER BY day;
