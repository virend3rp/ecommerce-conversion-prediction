-- ============================================================
-- 01_funnel_analysis.sql
-- E-commerce Conversion Funnel Analysis
-- Dataset: online_shoppers_intention
-- ============================================================

-- Overall conversion rate
SELECT
    COUNT(*) AS total_sessions,
    SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) AS converted_sessions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM online_shoppers;

-- ────────────────────────────────────────────────────────────
-- Conversion rate by traffic source
-- ────────────────────────────────────────────────────────────
SELECT
    TrafficType,
    COUNT(*) AS total_sessions,
    SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) AS conversions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct,
    ROUND(AVG(PageValues), 2) AS avg_page_value
FROM online_shoppers
GROUP BY TrafficType
ORDER BY conversion_rate_pct DESC;

-- ────────────────────────────────────────────────────────────
-- Conversion rate by visitor type
-- ────────────────────────────────────────────────────────────
SELECT
    VisitorType,
    COUNT(*) AS total_sessions,
    SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) AS conversions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct,
    ROUND(AVG(ProductRelated_Duration), 2) AS avg_product_time_sec
FROM online_shoppers
GROUP BY VisitorType
ORDER BY conversion_rate_pct DESC;

-- ────────────────────────────────────────────────────────────
-- Monthly conversion trend
-- ────────────────────────────────────────────────────────────
SELECT
    Month,
    COUNT(*) AS total_sessions,
    SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) AS conversions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM online_shoppers
GROUP BY Month
ORDER BY
    CASE Month
        WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3
        WHEN 'Apr' THEN 4 WHEN 'May' THEN 5 WHEN 'Jun' THEN 6
        WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8 WHEN 'Sep' THEN 9
        WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12
    END;

-- ────────────────────────────────────────────────────────────
-- Weekend vs Weekday conversion
-- ────────────────────────────────────────────────────────────
SELECT
    CASE WHEN Weekend = TRUE THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    COUNT(*) AS total_sessions,
    SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) AS conversions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM online_shoppers
GROUP BY Weekend;

-- ────────────────────────────────────────────────────────────
-- Bounce rate buckets vs conversion
-- ────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN BounceRates = 0 THEN '0% bounce'
        WHEN BounceRates <= 0.1 THEN '1–10%'
        WHEN BounceRates <= 0.25 THEN '11–25%'
        ELSE '25%+'
    END AS bounce_bucket,
    COUNT(*) AS sessions,
    ROUND(100.0 * SUM(CASE WHEN Revenue = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS conversion_rate_pct
FROM online_shoppers
GROUP BY bounce_bucket
ORDER BY conversion_rate_pct DESC;
