# Olist_Brazillian_Dataset

## 1) IMPORTING LIBRARIES, LOADING THE DATA AND BASIC OBSERVATIONS

import kagglehub

# Download latest version
path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

print("Path to dataset files:", path)

import numpy as np
import pandas as pd #Import pandas library
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import kruskal
import warnings
warnings.filterwarnings('ignore')
import sqlite3 as sql


from google.colab import drive
drive.mount('/content/drive')

customers = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv')
geolocation = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv')
order_items = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')
order_payments = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv')
orders = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')
products = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv')
sellers = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv')
product_cat_name = pd.read_csv('/kaggle/input/brazilian-ecommerce/product_category_name_translation.csv')

datasets = {
    "customers": customers,
    "geolocation": geolocation,
    "order_items": order_items,
    "order_payments": order_payments,
    "order_reviews": order_reviews,
    "orders": orders,
    "products": products,
    "sellers": sellers,
    "product_cat_name": product_cat_name
}


for name, df in datasets.items():
    print(f"\n{'='*40}\n Dataset: {name}\n{'='*40}")

    print("\nHEAD:")
    display(df.head())

    print("\nSHAPE:")
    display(df.shape)

    print("\nCOLUMNS:")
    display(df.columns)

    print("\nDATA TYPES:")
    display(df.dtypes)

    print("\nUNIQUE VALUES:")
    display(df.nunique())


    print("\nINFO:")
    display(df.info())  # Note: df.info() prints directly and returns None

    print("\nDESCRIBE (Numerical Columns):")
    display(df.describe())

    print("\nMISSING VALUES:")
    display(df.isnull().sum())

# Connect to an in-memory SQLite database
conn = sql.connect(':memory:')

# Import each dataset into a separate table in the SQLite database
for name, df in datasets.items():
    df.to_sql(name, conn, index=False, if_exists='replace')
    print(f"Imported {name} dataset into SQLite table '{name}'.")

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Function to print the number of rows and columns for a table
def print_table_info(table_name):
    # Get number of rows
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    num_rows = cursor.fetchone()[0]

    # Get number of columns
    # We can get column info from the `PRAGMA table_info` command
    cursor.execute(f"PRAGMA table_info({table_name});")
    num_cols = len(cursor.fetchall())

    print(f"Table '{table_name}': {num_rows} rows, {num_cols} columns.")

# Print row and column counts for each imported table
print("\nPrinting Row and Column Counts for each table:")
for name in datasets.keys():
    print_table_info(name)

# Close the connection (optional if you want to keep it open for further queries)
# conn.close()

# SQL AND DAV

# 1) Customer Lifetime Value Analysis




# Calculate the lifetime value of customers by summing total revenue per customer and analyzing it across different states and cities.


sql_query = """
SELECT
    c.customer_state,
    c.customer_city,
    SUM(op.payment_value) AS total_revenue,
    COUNT(DISTINCT c.customer_id) AS unique_customers,
    SUM(op.payment_value) / COUNT(DISTINCT c.customer_id) AS average_lifetime_value
FROM
    customers c
JOIN
    orders o ON c.customer_id = o.customer_id
JOIN
    order_payments op ON o.order_id = op.order_id
GROUP BY
    c.customer_state,
    c.customer_city
ORDER BY
    total_revenue DESC;
"""
customer_lifetime_value = pd.read_sql_query(sql_query, conn)
display(customer_lifetime_value.head())


plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)


sns.barplot(x='customer_state', y='total_revenue', data=customer_lifetime_value)
plt.title('Total Revenue by Customer State')
plt.xlabel('Customer State')
plt.ylabel('Total Revenue')
plt.xticks(rotation=55, ha='left')


plt.subplot(1,2,2)
# Plot a histogram of the average lifetime value on the second subplot
sns.histplot(customer_lifetime_value['average_lifetime_value'], bins=50, kde=True)
plt.title('Distribution of Average Lifetime Value by Customer Location')
plt.xlabel('Average Lifetime Value')
plt.ylabel('Frequency')

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()

# Show the plots
plt.show()

# **Insights for Customers Lifetime Value Analysis:**

1)The bar plot of 'Total Revenue by Customer State' clearly shows which states generate the most overall revenue.States with higher bars are crucial markets.

2) The granularity of the analysis by examining CLV at the zip code level for more detailed insights beyond state and city.

3) A skewed distribution might indicate some locations have significantly higher or lower average CLV compared to others.

# 2) Churn Detection: Inactive Customers

# Identify customers who haven't ordered anything in the last 6 months before the last available order date.


sql_query_inactive_customers = """
WITH LastOrderDate AS (
    SELECT MAX(order_purchase_timestamp) AS last_order_date
    FROM orders
),
CustomerLastOrder AS (
    SELECT
        customer_id,
        MAX(order_purchase_timestamp) AS last_purchase_date
    FROM orders
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state
FROM
    customers c
LEFT JOIN
    CustomerLastOrder cl ON c.customer_id = cl.customer_id
CROSS JOIN
    LastOrderDate lod
WHERE
    -- Case 1: Customer has no orders (last_purchase_date is NULL)
    -- Case 2: Customer's last order was more than 6 months before the overall last order date
    cl.last_purchase_date IS NULL OR STRFTIME('%Y-%m-%d %H:%M:%S', cl.last_purchase_date) < STRFTIME('%Y-%m-%d %H:%M:%S', lod.last_order_date, '-6 months');
"""

inactive_customers_df = pd.read_sql_query(sql_query_inactive_customers, conn)
display(inactive_customers_df.head(10))
print(f"Number of inactive customers: {len(inactive_customers_df)}")


plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
# Visualize the distribution of inactive customers by state
sns.countplot(data=inactive_customers_df, x='customer_state', order=inactive_customers_df['customer_state'].value_counts().index)
plt.title('Distribution of Inactive Customers by State')
plt.xlabel('State')
plt.ylabel('Number of Inactive Customers')
plt.xticks(rotation=45, ha='right')


plt.subplot(1,2,2)
# Visualize the top 10 cities with the most inactive customers
top_cities = inactive_customers_df['customer_city'].value_counts().nlargest(10).index
sns.countplot(data=inactive_customers_df[inactive_customers_df['customer_city'].isin(top_cities)], x='customer_city', order=top_cities)
plt.title('Top 10 Cities with Most Inactive Customers')
plt.xlabel('City')
plt.ylabel('Number of Inactive Customers')
plt.xticks(rotation=45, ha='right')
plt.show()

# **Insights based on the analysis of inactive customers: **

1) The analysis identifies a large number of customers who haven't placed an order in the last 6 months.

2) The count plots reveal that customer inactivity is not uniform across all states and cities.

3) Interestingly, the states with the highest number of inactive customers (like SP) often align with the states that have the highest total revenue and unique customers (as seen in the CLV analysis).

4) Understanding the geographic distribution can help tailor these campaigns based on regional characteristics or past purchasing behavior.



# 3)Delivery Delay Impact on Reviews

# Analyze how delivery delays affect review scores. Do late deliveries tend to get lower ratings?

sql_query_inactive_customers = """
SELECT
    CASE
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'Late'
        ELSE 'On Time'
    END AS delivery_status,
    AVG(CAST(r.review_score AS REAL)) AS average_review_score,
    COUNT(o.order_id) AS number_of_orders
FROM
    orders o
JOIN
    order_reviews r ON o.order_id = r.order_id
WHERE
    o.order_delivered_customer_date IS NOT NULL -- Only consider delivered orders
GROUP BY
    delivery_status;"""

delivery_delay_impact = pd.read_sql_query(sql_query_inactive_customers, conn)
display(delivery_delay_impact.head())


# Visualize the impact
plt.figure(figsize=(8, 5))
sns.barplot(x='delivery_status', y='average_review_score',hue='number_of_orders', data=delivery_delay_impact)
plt.title('Average Review Score by Delivery Status')
plt.xlabel('Delivery Status')
plt.ylabel('Average Review Score')
plt.ylim(0, 5) # Set y-axis limit for review scores
plt.show()




# Insights for Delivery Delay Impact on Reviews:


1) The bar plot clearly shows that orders delivered "Late" have a noticeably lower average review score compared to orders delivered "On Time".

2) The `delivery_delay_impact` DataFrame provides the exact average review scores.

3) The "number_of_orders" column (although not explicitly visualized on its own but potentially influencing the hue in the bar plot if that's how `hue='number_of_orders'` is interpreted by seaborn in this context) and the underlying data likely show that the vast majority of orders are delivered on time.

4) This analysis confirms that the delivery experience is a crucial factor influencing customer reviews.

5) The insights strongly suggest that minimizing delivery delays should be a top priority for improving overall customer satisfaction and potentially increasing average review scores.

# 4) Multi-payment Analysis per Order



# Which orders used more than one payment type, and how does this relate to their total order value

# prompt: generate sql Which orders used more than one payment type, and how does this relate to their total order value

sql_query_multi_payment = """
SELECT
    op.order_id,
    COUNT(DISTINCT op.payment_type) AS distinct_payment_types,
    SUM(op.payment_value) AS total_order_value
FROM
    order_payments op
GROUP BY
    op.order_id
HAVING
    COUNT(DISTINCT op.payment_type) > 1;
"""
multi_payment_orders = pd.read_sql_query(sql_query_multi_payment, conn)
display(multi_payment_orders.head())
print(f"Number of orders with more than one payment type: {len(multi_payment_orders)}")

# Visualize the relationship between the number of distinct payment types and total order value
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=multi_payment_orders, x='distinct_payment_types')
plt.title('Number of Distinct Payment Types per Order')
plt.xlabel('Number of Payment Types')
plt.ylabel('Number of Orders')

plt.subplot(1, 2, 2)
sns.violinplot(data=multi_payment_orders, x='distinct_payment_types', y='total_order_value')
plt.title('Total Order Value vs. Number of Payment Types')
plt.xlabel('Number of Payment Types')
plt.ylabel('Total Order Value')

plt.tight_layout()
plt.show()

**Insights from Multi-payment Analysis per Order:**

1. The count plot shows that orders utilizing more than one payment type are relatively infrequent compared to the total number of orders. The vast majority of orders likely use a single payment method.

2. The count plot also shows the distribution of orders based on the exact number of distinct payment types used (e.g., orders with 2 payment types, orders with 3 payment types, etc.).

3. The violin plot provides a key insight: orders with multiple payment types tend to have a higher total order value compared to orders with a single payment type (which are not explicitly shown in this analysis but can be inferred as the baseline). The violin shape suggests that the distribution of total order values for multi-payment orders is generally shifted towards higher values.

4. Customers making larger purchases might need the flexibility to split payments across different cards, use a combination of credit and debit, or utilize other available payment methods.


# 5) Product Recommendation Seed: Customers Who Bought X Also Bought Y


#Which combinations of product categories are most commonly bought together in a single order? List the top 3 most frequent category pairs.

sql_query_product_recommendation = """
SELECT
    pca.product_category_name_english AS category1,
    pcb.product_category_name_english AS category2,
    COUNT(DISTINCT oi_a.order_id) AS order_count
FROM
    order_items oi_a
JOIN
    order_items oi_b ON oi_a.order_id = oi_b.order_id AND oi_a.order_item_id != oi_b.order_item_id
JOIN
    products p_a ON oi_a.product_id = p_a.product_id
JOIN
    products p_b ON oi_b.product_id = p_b.product_id
JOIN
    product_cat_name pca ON p_a.product_category_name = pca.product_category_name
JOIN
    product_cat_name pcb ON p_b.product_category_name = pcb.product_category_name
WHERE
    pca.product_category_name_english < pcb.product_category_name_english -- To avoid duplicate pairs (e.g., A-B and B-A) and self-pairs (A-A)
GROUP BY
    pca.product_category_name_english,
    pcb.product_category_name_english
ORDER BY
    order_count DESC
LIMIT 3; """
product_recommendation = pd.read_sql_query(sql_query_product_recommendation, conn)
display(product_recommendation.head())


# Visualize the top 3 most frequent product category co-occurrences
plt.figure(figsize=(10, 6))
sns.barplot(
    y='order_count',
    x=product_recommendation['category1'] + ' & ' + product_recommendation['category2'], # Combine category names for y-axis labels
    data=product_recommendation,
    palette='viridis'
)
plt.title('Top 3 Most Frequent Product Category Co-occurrences in Orders')
plt.xlabel('Number of Orders')
plt.ylabel('Product Category Pair')
plt.show()

## **Insights from Product Recommendation Seed: Customers Who Bought X Also Bought Y**


1. The bar plot specifically highlight the top 3 most frequent category combinations. These pairs represent strong co-purchase patterns.

2. This simple co-occurrence analysis can power "Frequently Bought Together" sections on product pages or serve as input for more complex collaborative filtering algorithms.

3. Ensuring sufficient stock of items within frequently paired categories can help fulfill customer demand and avoid lost sales.

4. The Potential for Category-Specific Strategies could lead to exploring why these specific combinations are popular and developing tailored strategies for those categories.

# 6) Delivery Time Deviation from SLA



# For each seller, calculate the average difference between the actual delivery date and the estimated delivery date, and identify sellers who consistently deliver late

sql_query_delivery_deviation="""
WITH delivery_delays AS (
    SELECT
        oi.seller_id,
        o.order_id,
        DATE(o.order_delivered_customer_date) AS actual_delivery,
        DATE(o.order_estimated_delivery_date) AS estimated_delivery,
        JULIANDAY(o.order_delivered_customer_date) - JULIANDAY(o.order_estimated_delivery_date) AS delay_in_days
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE
        o.order_delivered_customer_date IS NOT NULL
        AND o.order_estimated_delivery_date IS NOT NULL
),
seller_delay_avg AS (
    SELECT
        seller_id,
        ROUND(AVG(delay_in_days), 2) AS avg_delay
    FROM delivery_delays
    WHERE delay_in_days > 0 -- Only delays
    GROUP BY seller_id
)
SELECT * FROM seller_delay_avg
ORDER BY avg_delay DESC
LIMIT 10;
"""
delivery_deviation = pd.read_sql_query(sql_query_delivery_deviation, conn)
display(delivery_deviation.head())

plt.figure(figsize=(12, 6))
sns.barplot(x='seller_id', y='avg_delay', data=delivery_deviation, palette='Reds_d')

plt.title('Top 10 Sellers with Highest Average Delivery Delay')
plt.xlabel('Seller ID')
plt.ylabel('Average Delay (Days)')
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()


# **Insights from Delivery Time Deviation from SLA:**

1. The analysis successfully identifies sellers who consistently exhibit the largest average delays beyond the estimated delivery date.

2. A positive value indicates the average number of days a seller's deliveries are late, on average, when they are delayed.

3. Investigating the reasons behind delays for these specific sellers (e.g., logistics issues, processing time, inventory problems) is crucial for improvement.

4. Sellers with higher average delays are likely contributing significantly to negative customer experiences and lower review scores.

# 7. Product Category Margin Analysis

# Find the product categories with the highest average profit margin, where profit margin is calculated as (price - freight value).

sql_query_product_category_margin = """
SELECT
    pc.product_category_name_english,
    AVG(oi.price - oi.freight_value) AS average_profit_margin,
    COUNT(oi.order_item_id) AS total_items_sold
FROM
    order_items oi
JOIN
    products p ON oi.product_id = p.product_id
LEFT JOIN
    product_cat_name pc ON p.product_category_name = pc.product_category_name
GROUP BY
    pc.product_category_name_english
ORDER BY
    average_profit_margin DESC
LIMIT 15; -- Display top 10 categories by average margin
"""

product_category_margin = pd.read_sql_query(sql_query_product_category_margin, conn)
display(product_category_margin.head(15))


# Visualize the top 10 product categories by average profit margin
plt.figure(figsize=(12, 7))
sns.barplot(y='average_profit_margin', x='product_category_name_english',hue='total_items_sold', data=product_category_margin.head(15), palette='coolwarm')
plt.title('Top 10 Product Categories by Average Profit Margin')
plt.ylabel('Average Profit Margin')
plt.xlabel('Product Category (English)')
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

# **Insights from Product Category Margin Analysis:**

1. The `product_category_margin` DataFrame and the bar plot highlight the categories with the highest average margins.

2. This allows for an important comparison: some categories might have a very high average margin but low sales volume, while others might have a moderate margin but very high volume. Both types are important, but they require different strategic considerations.

3. The categories with the highest average profit margins are critical for overall profitability.

4. True profit margin would also include other costs like the cost of goods sold (COGS), operational expenses, marketing costs, etc.

# 8. Seller Performance Index

# Create a composite score for each seller based on delivery speed, customer reviews, and return rate.

sql_query_seller_performance_index = """
WITH SellerDelivery AS (
    SELECT
        oi.seller_id,
        AVG(JULIANDAY(o.order_delivered_customer_date) - JULIANDAY(o.order_estimated_delivery_date)) AS average_delivery_delay_days,
        COUNT(o.order_id) AS total_delivered_orders
    FROM
        orders o
    JOIN
        order_items oi ON o.order_id = oi.order_id
    WHERE
        o.order_delivered_customer_date IS NOT NULL
        AND o.order_estimated_delivery_date IS NOT NULL
    GROUP BY
        oi.seller_id
),
SellerReviews AS (
    SELECT
        oi.seller_id,
        AVG(CAST(r.review_score AS REAL)) AS average_review_score,
        COUNT(r.review_id) AS total_reviews
    FROM
        order_items oi
    JOIN
        order_reviews r ON oi.order_id = r.order_id
    GROUP BY
        oi.seller_id
),
SellerReturns AS (
    -- This is a simplified approach for return rate.
    -- A more robust method might involve tracking returned items explicitly if available.
    -- Here, we'll approximate using cancelled orders linked to a seller.
    -- NOTE: This is an approximation as cancellation doesn't equal return.
    -- A proper return rate requires item-level return data.
    SELECT
        oi.seller_id,
        CAST(SUM(CASE WHEN o.order_status = 'canceled' THEN 1 ELSE 0 END) AS REAL) * 100.0 / COUNT(o.order_id) AS cancellation_rate -- Using cancellation as a proxy for return rate
    FROM
        order_items oi
    JOIN
        orders o ON oi.order_id = o.order_id
    GROUP BY
        oi.seller_id
)
-- Combine metrics and create a composite score
SELECT
    s.seller_id,
    sd.average_delivery_delay_days,
    sr.average_review_score,
    str.cancellation_rate, -- Using cancellation rate as the return rate proxy
    -- Composite Score Calculation (Example)
    -- Inverse of delay (lower is better), multiplied by average review score (higher is better),
    -- minus a factor for cancellation rate (lower is better).
    -- Scaling the values before combining might be necessary for a balanced score.
    -- This is a basic example; weights and normalization would be needed for a real index.
    (5.0 - COALESCE(sd.average_delivery_delay_days, 0.0)) -- Assuming maximum delay penalty caps at 5
    + COALESCE(sr.average_review_score, 0.0)
    - (COALESCE(str.cancellation_rate, 0.0) / 10.0) -- Simple scaling for cancellation rate
    AS composite_score
FROM
    sellers s
LEFT JOIN SellerDelivery sd ON s.seller_id = sd.seller_id
LEFT JOIN SellerReviews sr ON s.seller_id = sr.seller_id
LEFT JOIN SellerReturns str ON s.seller_id = str.seller_id
ORDER BY
    composite_score DESC;"""
seller_performance_index = pd.read_sql_query(sql_query_seller_performance_index, conn)
display(seller_performance_index.head(10))


# prompt: # prompt: # Visualize top 10 sellers by composite score withn inverted x axis

import matplotlib.pyplot as plt
# Visualize top 10 sellers by composite score with inverted x axis
top_10_sellers = seller_performance_index.head(10).sort_values(by='composite_score', ascending=True) # Sort ascending for inverted x-axis

plt.figure(figsize=(10, 6))
sns.barplot(y='seller_id', x='composite_score', data=top_10_sellers, palette='viridis') # Use y for seller_id to invert x
plt.title('Top 10 Sellers by Composite Score')
plt.xlabel('Composite Score')
plt.ylabel('Seller ID')
plt.tight_layout()
plt.show()

# Visualize top 10 sellers by composite score
if not seller_performance_index.empty:
    top_performing_sellers = seller_performance_index.head(10)
    plt.figure(figsize=(10, 7))
    sns.barplot(
        y='composite_score',
        x='seller_id',
        data=top_performing_sellers,
        palette='viridis'
    )
    plt.title('Top 10 Sellers by Composite Performance Score')
    plt.ylabel('Composite Score')
    plt.xlabel('Seller ID')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo seller performance data available to display.")


## **Insights from Seller Performance Index:**

1. The analysis successfully creates a composite score that attempts to rank sellers based on a combination of delivery speed, review scores, and a proxy for return rate (cancellation rate).

2. By ordering the results, you can easily identify the sellers with the highest composite scores (top performers) and, by looking at the bottom of the sorted list (or sorting ascending), the sellers with the lowest scores (those needing the most attention).

3. The individual metrics within the composite score (average delay, average review score, cancellation rate) are displayed alongside the composite score. This allows you to see the specific strengths and weaknesses contributing to a seller's overall score.

4. The composite index provides a single, albeit simplified, metric for performance comparison.

# 9. Repeat Purchase Patterns

# What % of customers place a second order within 30, 60, and 90 days?

sql_query_repeat_purchase_rate = """
WITH CustomerFirstOrder AS (
    SELECT
        c.customer_unique_id,
        MIN(o.order_purchase_timestamp) AS first_order_date
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id -- Join with customers table
    GROUP BY c.customer_unique_id
),
CustomerSubsequentOrders AS (
    SELECT
        c.customer_unique_id,
        o.order_purchase_timestamp AS subsequent_order_date,
        cf.first_order_date,
        JULIANDAY(o.order_purchase_timestamp) - JULIANDAY(cf.first_order_date) AS days_since_first_order
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id -- Join with customers table
    JOIN CustomerFirstOrder cf ON c.customer_unique_id = cf.customer_unique_id
    WHERE o.order_purchase_timestamp > cf.first_order_date -- Only consider orders after the first one
),
RepeatCustomersWithinPeriods AS (
    SELECT DISTINCT
        customer_unique_id,
        CASE
            WHEN days_since_first_order <= 30 THEN 1 ELSE 0
        END AS repeat_30_days,
        CASE
            WHEN days_since_first_order <= 60 THEN 1 ELSE 0
        END AS repeat_60_days,
        CASE
            WHEN days_since_first_order <= 90 THEN 1 ELSE 0
        END AS repeat_90_days
    FROM CustomerSubsequentOrders
)
SELECT
    CAST(SUM(repeat_30_days) AS REAL) * 100.0 / (SELECT COUNT(DISTINCT customer_unique_id) FROM customers) AS repeat_rate_30_days_percent,
    CAST(SUM(repeat_60_days) AS REAL) * 100.0 / (SELECT COUNT(DISTINCT customer_unique_id) FROM customers) AS repeat_rate_60_days_percent,
    CAST(SUM(repeat_90_days) AS REAL) * 100.0 / (SELECT COUNT(DISTINCT customer_unique_id) FROM customers) AS repeat_rate_90_days_percent
FROM RepeatCustomersWithinPeriods;
"""

repeat_purchase_rate_df = pd.read_sql_query(sql_query_repeat_purchase_rate, conn)
display(repeat_purchase_rate_df)

# Create a DataFrame for plotting
repeat_rates = pd.DataFrame({
    'Period (Days)': ['30 Days', '60 Days', '90 Days'],
    'Repeat Customer Rate (%)': [repeat_purchase_rate_df['repeat_rate_30_days_percent'].iloc[0],
                                 repeat_purchase_rate_df['repeat_rate_60_days_percent'].iloc[0],
                                 repeat_purchase_rate_df['repeat_rate_90_days_percent'].iloc[0]]
})

# Visualize the repeat purchase rates over time
plt.figure(figsize=(8, 5))
sns.barplot(x='Period (Days)', y='Repeat Customer Rate (%)', data=repeat_rates, palette='viridis')
plt.title('Percentage of Customers Placing a Second Order Within Different Time Periods')
plt.xlabel('Time Period After First Order')
plt.ylabel('Repeat Customer Rate (%)')
plt.ylim(0, repeat_rates['Repeat Customer Rate (%)'].max() * 1.1) # Set y-axis limit for better visualization
plt.show()



# **Insights derived from the Repeat Purchase Patterns Analysis:**

1. The most striking insight is likely the low percentage of customers who place a second order within 30, 60, and 90 days.The exact percentages from your `repeat_purchase_rate_df` will quantify this precisely.

2. As the time window increases from 30 to 60 to 90 days, the percentage of repeat customers naturally increases.

3. A low repeat purchase rate indicates a significant opportunity to focus on customer retention.

4. A seemingly low rate might be typical for this type of e-commerce model, or it could indicate a significant area for improvement.

5. Making the first purchase smooth, positive, and memorable is key to increasing the likelihood of a repeat order.

# 10. Cross-category Purchase Behavior

# Identify customers who purchased from 3 or more different product categories.

---



sql_query_cross_category_customers = """
WITH CustomerCategoryPurchases AS (
    SELECT
        c.customer_unique_id,
        p.product_category_name,
        COUNT(DISTINCT oi.order_id) AS num_orders_in_category -- Count unique orders per category for a customer
    FROM
        customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE p.product_category_name IS NOT NULL -- Exclude items with missing category
    GROUP BY
        c.customer_unique_id,
        p.product_category_name
),
CustomerCategoryCount AS (
    SELECT
        customer_unique_id,
        COUNT(DISTINCT product_category_name) AS distinct_category_count
    FROM
        CustomerCategoryPurchases
    GROUP BY
        customer_unique_id
)
SELECT
    cc.customer_unique_id,
    cc.distinct_category_count
FROM
    CustomerCategoryCount cc
WHERE
    cc.distinct_category_count >= 3
ORDER BY
    cc.distinct_category_count DESC;
"""

cross_category_customers = pd.read_sql_query(sql_query_cross_category_customers, conn)
display(cross_category_customers.head(10))
print(f"Number of customers who purchased from 3 or more different categories: {len(cross_category_customers)}")

# Visualize the distribution of the number of distinct categories purchased for customers with >= 3 categories
plt.figure(figsize=(8, 6))
sns.histplot(cross_category_customers['distinct_category_count'], bins=range(3, cross_category_customers['distinct_category_count'].max() + 2), kde=False, discrete=True)
plt.title('Distribution of Number of Distinct Product Categories Purchased (Customers with >= 3 Categories)')
plt.xlabel('Number of Distinct Product Categories')
plt.ylabel('Number of Customers')
plt.xticks(range(3, cross_category_customers['distinct_category_count'].max() + 1)) # Set x-ticks to integer category counts
plt.grid(axis='y', linestyle='--')
plt.show()



# **Insights from Cross-category Purchase Behavior:**

1. The analysis successfully identifies customers who exhibit diverse purchasing behavior by buying from three or more distinct product categories.

2. The count of `cross_category_customers` provides a clear number of how many customers fall into this "diverse buyer" segment.

3. Customers who purchase across multiple categories are often highly engaged and potentially high-value customers.

4. By understanding the categories they've already purchased from, the platform can recommend products from other relevant categories, increasing the likelihood of additional purchases.



# **Top Selling Products Categories**

#11.  What are the top 10 best-selling product categories by revenue and by order count, and how do their sales volumes compare?


sql_query_top_categories = """
SELECT
    pc.product_category_name_english,
    SUM(oi.price + oi.freight_value) AS total_revenue,
    COUNT(DISTINCT oi.order_id) AS total_orders
FROM
    order_items oi
JOIN
    products p ON oi.product_id = p.product_id
LEFT JOIN
    product_cat_name pc ON p.product_category_name = pc.product_category_name
WHERE
    pc.product_category_name_english IS NOT NULL -- Exclude items with unknown categories
GROUP BY
    pc.product_category_name_english
ORDER BY
    total_revenue DESC; -- Order by revenue first
"""

top_categories_df = pd.read_sql_query(sql_query_top_categories, conn)
display(top_categories_df.head(10))

# Create subplots for revenue and order count
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plotting Top 10 by Revenue
sns.barplot(
    x='total_revenue',
    y='product_category_name_english',
    data=top_categories_df.head(10),
    palette='viridis',
    ax=axes[0]
)
axes[0].set_title('Top 10 Product Categories by Total Revenue')
axes[0].set_xlabel('Total Revenue')
axes[0].set_ylabel('Product Category (English)')
axes[0].invert_yaxis() # Invert y-axis to have the highest value at the top


# Plotting Top 10 by Order Count
sns.barplot(
    x='total_orders',
    y='product_category_name_english',
    data=top_categories_df.sort_values(by='total_orders',ascending=False).head(10),
    palette='plasma',
    ax=axes[1]
)
axes[1].set_title('Top 10 Product Categories by Order Count')
axes[1].set_xlabel('Total Order Count')
axes[1].invert_yaxis() # Invert y-axis to have the highest value at the top


plt.tight_layout()
plt.show()


**Insights from Top Selling Product Categories Analysis:**

1. The analysis clearly identifies the product categories that generate the most revenue and those that are most frequently purchased (highest order count). .

2. Some categories might rank high in revenue but lower in order count, indicating higher average price points.

3. The top-performing categories by both metrics are likely core to the business and warrant significant strategic focus: **Inventory Management,** **Marketing and Promotion,** **Product Development,** **Pricing Strategies.**

4. While the analysis is at the top-level category, a deeper dive could analyze subcategories within the top performers to identify the most successful product types within those broad categories.

## PROBABILITY & STATISTIC

## What is the probability that a customer is inactive given that they are from the state of Sao Paulo (SP)?

- Event A: Customer is inactive (based on our previous definition)
- Event B: Customer is from the state of SP

- P(A) = Probability of a customer being inactive
- P(B) = Probability of a customer being from SP
- P(A|B) = Probability of a customer being inactive given they are from SP
- P(B|A) = Probability of a customer being from SP given they are inactive
- Find P(A|B)

# First, let's get the total number of unique customers
total_unique_customers_query = """
SELECT COUNT(DISTINCT customer_unique_id) FROM customers;
"""
total_unique_customers = pd.read_sql_query(total_unique_customers_query, conn).iloc[0, 0]
display(total_unique_customers)


# We already have the inactive customers in `inactive_customers_df`
num_inactive_customers = len(inactive_customers_df)
num_inactive_customers

# Get the total number of unique customers from SP
customers_from_sp_query = """
SELECT COUNT(DISTINCT customer_unique_id) FROM customers WHERE customer_state = 'SP';
"""
num_customers_from_sp = pd.read_sql_query(customers_from_sp_query, conn).iloc[0, 0]
display(num_customers_from_sp)

# P(A) = Number of inactive customers / Total unique customers
p_inactive = num_inactive_customers / total_unique_customers
print(f"P(Inactive): {p_inactive:.4f}")
# P(B) = Number of unique customers from SP / Total unique customers
p_sp = num_customers_from_sp / total_unique_customers
print(f"P(SP): {p_sp:.4f}")

# Now, find the number of customers who are BOTH inactive AND from SP (Event A and B)
inactive_customers_from_sp = inactive_customers_df[inactive_customers_df['customer_state'] == 'SP']
num_inactive_and_sp = len(inactive_customers_from_sp)
print("num_inactive_and_sp :",num_inactive_and_sp)

# P(A and B) = Number of inactive customers from SP / Total unique customers
p_inactive_and_sp = num_inactive_and_sp / total_unique_customers
print(f"P(Inactive and SP): {p_inactive_and_sp:.4f}")

# Calculate Conditional Probability P(A|B) = P(Inactive | SP)
# P(Inactive | SP) = P(Inactive and SP) / P(SP)

if p_sp > 0:
    p_inactive_given_sp = p_inactive_and_sp / p_sp
    print(f"P(Inactive | SP) = P(Inactive and SP) / P(SP) = {p_inactive_and_sp:.4f} / {p_sp:.4f} = {p_inactive_given_sp:.4f}")
else:
    p_inactive_given_sp = 0
    print("P(SP) is 0, cannot calculate P(Inactive | SP)")

# Now, let's use Bayes' Theorem to calculate P(A|B)
# P(A|B) = P(B|A) * P(A) / P(B)

# First, calculate P(B|A) = Probability of being from SP given the customer is inactive
# P(B|A) = P(SP | Inactive) = Number of inactive customers from SP / Total number of inactive customers
if num_inactive_customers > 0:
    p_sp_given_inactive = num_inactive_and_sp / num_inactive_customers
    print(f"P(SP | Inactive) = Number of inactive customers from SP / Total inactive customers = {num_inactive_and_sp} / {num_inactive_customers} = {p_sp_given_inactive:.4f}")
else:
    p_sp_given_inactive = 0
    print("Number of inactive customers is 0, cannot calculate P(SP | Inactive)")

# Now apply Bayes' Theorem
# P(Inactive | SP) = P(SP | Inactive) * P(Inactive) / P(SP)
if p_sp > 0 and p_inactive > 0:
    bayesian_p_inactive_given_sp = p_sp_given_inactive * p_inactive / p_sp
    print(f"Bayes' Theorem Result for P(Inactive | SP): {bayesian_p_inactive_given_sp:.4f}")
    # Verify that the conditional probability matches the Bayes' result (they should be the same)
    print(f"Conditional Probability Result: {p_inactive_given_sp:.4f}")
else:
    print("Cannot apply Bayes' Theorem due to zero probabilities.")

## What is the probability of a delivery being Late given that the review score is 1?

- Event A: Delivery is Late
- Event B: Review score is 1

- P(A) = Probability of Late delivery
- P(B) = Probability of Review Score 1
- P(A and B) = Probability of Late delivery AND Review Score 1
- P(A|B) = P(Late | Review Score 1)
- P(B|A) = P(Review Score 1 | Late) = Number of Late deliveries with Review Score 1 / Total number of Late deliveries

# Reconnect to the in-memory SQLite database
conn = sql.connect(':memory:')

# Re-import the datasets into the database
for name, df in datasets.items():
    df.to_sql(name, conn, index=False, if_exists='replace')

# Total number of delivered orders with reviews
total_delivered_orders_with_reviews_query = """
SELECT COUNT(o.order_id)
FROM orders o
JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_delivered_customer_date IS NOT NULL;
"""
total_delivered_orders_with_reviews = pd.read_sql_query(total_delivered_orders_with_reviews_query, conn).iloc[0, 0]
print(f"\nTotal delivered orders with reviews: {total_delivered_orders_with_reviews}")

# Number of Late deliveries
num_late_deliveries_query = """
SELECT COUNT(o.order_id)
FROM orders o
JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_delivered_customer_date IS NOT NULL
AND o.order_delivered_customer_date > o.order_estimated_delivery_date;
"""
num_late_deliveries = pd.read_sql_query(num_late_deliveries_query, conn).iloc[0, 0]
print(f"Number of Late deliveries (with reviews): {num_late_deliveries}")

# Number of orders with Review Score 1
num_review_score_1_query = """
SELECT COUNT(r.review_id)
FROM order_reviews r
JOIN orders o ON r.order_id = o.order_id -- Ensure the order was delivered
WHERE r.review_score = 1
AND o.order_delivered_customer_date IS NOT NULL;
"""
num_review_score_1 = pd.read_sql_query(num_review_score_1_query, conn).iloc[0, 0]
print(f"Number of orders with Review Score 1 (delivered): {num_review_score_1}")

# Number of Late deliveries with Review Score 1 (Event A and B)
num_late_and_review_1_query = """
SELECT COUNT(o.order_id)
FROM orders o
JOIN order_reviews r ON o.order_id = r.order_id
WHERE o.order_delivered_customer_date IS NOT NULL
AND o.order_delivered_customer_date > o.order_estimated_delivery_date
AND r.review_score = 1;
"""
num_late_and_review_1 = pd.read_sql_query(num_late_and_review_1_query, conn).iloc[0, 0]
print(f"Number of Late deliveries with Review Score 1: {num_late_and_review_1}")

# P(A) = Probability of Late delivery
p_late = num_late_deliveries / total_delivered_orders_with_reviews
print(f"P(Late delivery): {p_late:.4f}")

# P(B) = Probability of Review Score 1
p_review_1 = num_review_score_1 / total_delivered_orders_with_reviews
print(f"P(Review Score 1): {p_review_1:.4f}")

# P(A and B) = Probability of Late delivery AND Review Score 1
p_late_and_review_1 = num_late_and_review_1 / total_delivered_orders_with_reviews
print(f"P(Late delivery and Review Score 1): {p_late_and_review_1:.4f}")

# Calculate Conditional Probability P(A|B) = P(Late | Review Score 1)
# P(Late | Review Score 1) = P(Late and Review Score 1) / P(Review Score 1)
if p_review_1 > 0:
    p_late_given_review_1 = p_late_and_review_1 / p_review_1
    print(f"P(Late | Review Score 1) = P(Late and Review Score 1) / P(Review Score 1) = {p_late_and_review_1:.4f} / {p_review_1:.4f} = {p_late_given_review_1:.4f}")
else:
    p_late_given_review_1 = 0
    print("P(Review Score 1) is 0, cannot calculate P(Late | Review Score 1)")

# Now, calculate P(B|A) = Probability of Review Score 1 given the delivery was Late
# P(B|A) = P(Review Score 1 | Late) = Number of Late deliveries with Review Score 1 / Total number of Late deliveries
if num_late_deliveries > 0:
    p_review_1_given_late = num_late_and_review_1 / num_late_deliveries
    print(f"P(Review Score 1 | Late) = Number of Late deliveries with Review Score 1 / Total Late deliveries = {num_late_and_review_1} / {num_late_deliveries} = {p_review_1_given_late:.4f}")
else:
    p_review_1_given_late = 0
    print("Number of Late deliveries is 0, cannot calculate P(Review Score 1 | Late)")


# Apply Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
# P(Late | Review Score 1) = P(Review Score 1 | Late) * P(Late) / P(Review Score 1)
if p_review_1 > 0 and p_late > 0:
    bayesian_p_late_given_review_1 = p_review_1_given_late * p_late / p_review_1
    print(f"Bayes' Theorem Result for P(Late | Review Score 1): {bayesian_p_late_given_review_1:.4f}")
    # Verify that the conditional probability matches the Bayes' result
    print(f"Conditional Probability Result: {p_late_given_review_1:.4f}")
else:
     print("Cannot apply Bayes' Theorem due to zero probabilities.")

##  Assuming order prices follow approximately a normal distribution, what is the probability that a randomly selected order item has a price between $50 and $150?

# Note: We will first check if the price distribution is roughly normal and then calculate the probability.
# This requires calculating mean and standard deviation of order item prices.

# Get all order item prices
order_item_prices_query = """
SELECT price FROM order_items;
"""
order_item_prices = pd.read_sql_query(order_item_prices_query, conn)['price'].dropna() # Drop potential NaN values

# Check if there are enough data points and if the data exists
if len(order_item_prices) < 2:
    print("\nNot enough order item price data to calculate probability based on normal distribution assumption.")
else:
    # Calculate mean and standard deviation
    mean_price = order_item_prices.mean()
    std_dev_price = order_item_prices.std()

    print(f"\nOrder item price mean: {mean_price:.2f}")
    print(f"Order item price standard deviation: {std_dev_price:.2f}")

    # Check for normality assumption (optional but good practice)
    # A simple way is to look at a histogram and check skewness/kurtosis
    plt.figure(figsize=(8, 5))
    sns.histplot(order_item_prices, bins=50, kde=True)
    plt.title('Distribution of Order Item Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate Z-scores for the bounds ($50 and $150)
    z_score_50 = (50 - mean_price) / std_dev_price
    z_score_150 = (150 - mean_price) / std_dev_price

    print(f"Z-score for $50: {z_score_50:.4f}")
    print(f"Z-score for $150: {z_score_150:.4f}")

    # Calculate the probability using the cumulative distribution function (CDF) of the normal distribution
    # P(50 < Price < 150) = P(Z < Z_150) - P(Z < Z_50)
    prob_between_50_and_150 = stats.norm.cdf(z_score_150) - stats.norm.cdf(z_score_50)

    print(f"\nAssuming normal distribution, the probability that an order item price is between $50 and $150 is: {prob_between_50_and_150:.4f}")

- The original distribution of order item prices is highly skewed, not normal.
- A common transformation to make skewed data more normal is the log transformation.
- Let's apply a log transformation (log1p, which is log(1+x) to handle potential zero values, although prices are likely non-zero).

# Apply log transformation to the prices
log_transformed_prices = np.log1p(order_item_prices)

# Check the distribution after log transformation
plt.figure(figsize=(8, 5))
sns.histplot(log_transformed_prices, bins=50, kde=True)
plt.title('Distribution of Log-Transformed Order Item Prices')
plt.xlabel('Log(Price + 1)')
plt.ylabel('Frequency')
plt.show()

# Calculate mean and standard deviation of log-transformed prices
mean_log_price = log_transformed_prices.mean()
std_dev_log_price = log_transformed_prices.std()

print(f"\nLog-transformed price mean: {mean_log_price:.4f}")
print(f"Log-transformed price standard deviation: {std_dev_log_price:.4f}")

# We want the probability that the original price is between $50 and $150.
# This is equivalent to the probability that the log-transformed price is between log(50+1) and log(150+1).
log_50 = np.log1p(50)
log_150 = np.log1p(150)

print(f"Log(50+1): {log_50:.4f}")
print(f"Log(150+1): {log_150:.4f}")

# Calculate Z-scores for the log-transformed bounds
z_score_log_50 = (log_50 - mean_log_price) / std_dev_log_price
z_score_log_150 = (log_150 - mean_log_price) / std_dev_log_price

print(f"Z-score for Log(51): {z_score_log_50:.4f}")
print(f"Z-score for Log(151): {z_score_log_150:.4f}")

# Calculate the probability using the CDF of the normal distribution on the Z-scores
# P(log(51) < Log(Price+1) < log(151)) = P(Z < Z_log_150) - P(Z < Z_log_50)
prob_between_log_50_and_log_150 = stats.norm.cdf(z_score_log_150) - stats.norm.cdf(z_score_log_50)

print(f"\nAssuming log-normal distribution for original prices, the probability that an order item price is between $50 and $150 is: {prob_between_log_50_and_log_150:.4f}")

## What is the conditional probability that a delivered order has multiple payment types (more than 1 distinct type) given that its total value is greater than $1000?

- Event A: Order has multiple payment types
- Event B: Order total value is > $1000



*   Find P(A|B) = P(Multiple Payments | Total Value > $1000)
=> P(Multiple Payments AND Total Value > $1000) / P(Total Value > $1000)




# Calculate the number of delivered orders with total value > $1000
orders_value_gt_1000_query = """
SELECT COUNT(o.order_id)
FROM orders o
JOIN (SELECT order_id, SUM(payment_value) as total_value FROM order_payments GROUP BY order_id) op ON o.order_id = op.order_id
WHERE o.order_delivered_customer_date IS NOT NULL
AND op.total_value > 1000;
"""
num_orders_value_gt_1000 = pd.read_sql_query(orders_value_gt_1000_query, conn).iloc[0, 0]
print(f"\nNumber of delivered orders with total value > $1000: {num_orders_value_gt_1000}")

# Calculate the probability of an order having total value > $1000 (among delivered orders with payments)
# Need total delivered orders with payment value information
total_delivered_orders_with_payments_query = """
SELECT COUNT(DISTINCT o.order_id)
FROM orders o
JOIN order_payments op ON o.order_id = op.order_id
WHERE o.order_delivered_customer_date IS NOT NULL;
"""
total_delivered_orders_with_payments = pd.read_sql_query(total_delivered_orders_with_payments_query, conn).iloc[0, 0]
print(f"Total delivered orders with payment info: {total_delivered_orders_with_payments}")

if total_delivered_orders_with_payments > 0:
    p_value_gt_1000 = num_orders_value_gt_1000 / total_delivered_orders_with_payments
    print(f"P(Total Value > $1000 | Delivered Order): {p_value_gt_1000:.4f}")
else:
    p_value_gt_1000 = 0
    print("No delivered orders with payment info available.")

# Calculate the number of delivered orders with BOTH multiple payment types AND total value > $1000
multi_payment_and_value_gt_1000_query = """
SELECT COUNT(o.order_id)
FROM orders o
JOIN (
    SELECT order_id, COUNT(DISTINCT payment_type) as distinct_payment_types, SUM(payment_value) as total_value
    FROM order_payments
    GROUP BY order_id
) op ON o.order_id = op.order_id
WHERE o.order_delivered_customer_date IS NOT NULL
AND op.distinct_payment_types > 1
AND op.total_value > 1000;
"""
num_multi_payment_and_value_gt_1000 = pd.read_sql_query(multi_payment_and_value_gt_1000_query, conn).iloc[0, 0]
print(f"Number of delivered orders with multiple payments AND total value > $1000: {num_multi_payment_and_value_gt_1000}")

# Calculate P(A and B) = P(Multiple Payments AND Total Value > $1000 | Delivered Order)
if total_delivered_orders_with_payments > 0:
    p_multi_payment_and_value_gt_1000 = num_multi_payment_and_value_gt_1000 / total_delivered_orders_with_payments
    print(f"P(Multiple Payments AND Total Value > $1000 | Delivered Order): {p_multi_payment_and_value_gt_1000:.4f}")
else:
    p_multi_payment_and_value_gt_1000 = 0
    print("No delivered orders with payment info available.")

# Calculate P(A|B) = P(Multiple Payments | Total Value > $1000)
if p_value_gt_1000 > 0:
    p_multi_payment_given_value_gt_1000 = p_multi_payment_and_value_gt_1000 / p_value_gt_1000
    print(f"\nP(Multiple Payments | Total Value > $1000) = {p_multi_payment_given_value_gt_1000:.4f}")
else:
    print("\nP(Total Value > $1000) is 0, cannot calculate P(Multiple Payments | Total Value > $1000).")

# **Insights on the conditional probability that a delivered order has multiple payment types (more than 1 distinct type) given that its total value is greater than $1000:**

In summary, a high value for P(Multiple Payments | Total Value > $1000) empirically supports the idea that offering multiple payment options is particularly important for facilitating and potentially encouraging high-value purchases.

## Assuming the number of orders per day follows a Poisson distribution, what is the probability of having exactly 100 orders tomorrow?

- We need to estimate the average number of orders per day (lambda) from the historical data.
- Use the `orders` dataset and `order_purchase_timestamp`.

# Convert timestamp to datetime objects
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

# Extract the date
orders['order_purchase_date'] = orders['order_purchase_timestamp'].dt.date

# Count orders per day
orders_per_day = orders.groupby('order_purchase_date').size().reset_index(name='order_count')

# Calculate the average number of orders per day (lambda)
if not orders_per_day.empty:
    lambda_orders_per_day = orders_per_day['order_count'].mean()
    print(f"\nEstimated average number of orders per day (lambda): {lambda_orders_per_day:.2f}")

    # Assuming Poisson distribution with lambda, calculate the probability of having exactly 100 orders
    # P(X=k) = (lambda^k * e^(-lambda)) / k!
    k = 100
    poisson_prob_100_orders = stats.poisson.pmf(k, lambda_orders_per_day)

    print(f"Assuming Poisson distribution with lambda={lambda_orders_per_day:.2f}, the probability of having exactly {k} orders tomorrow is: {poisson_prob_100_orders:.8f}")

else:
    print("\nNo order data available to estimate average daily orders for Poisson distribution calculation.")


**Hypothesis Testing**

### 1. Delivery Delay Impact on Review Scores

- Null Hypothesis (H0): The average review score is the same for on-time and late deliveries (μ_on_time = μ_late)
- Alternative Hypothesis (H1): The average review score is different for on-time and late deliveries (μ_on_time ≠ μ_late)

# Ensure orders and order_reviews are loaded and preprocessed for date columns
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

# Merge orders and reviews
orders_reviews = orders.merge(order_reviews[['order_id', 'review_score']], on='order_id', how='inner')

# Filter null values for delivered orders
delivered_orders_reviews = orders_reviews.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date', 'review_score'])

# Define delivery status
delivered_orders_reviews['delivery_status'] = delivered_orders_reviews.apply(
    lambda row: 'Late' if row['order_delivered_customer_date'] > row['order_estimated_delivery_date'] else 'On Time',
    axis=1
)

# Separate review scores by delivery status
on_time_reviews = delivered_orders_reviews[delivered_orders_reviews['delivery_status'] == 'On Time']['review_score'].astype(float)
late_reviews = delivered_orders_reviews[delivered_orders_reviews['delivery_status'] == 'Late']['review_score'].astype(float)

# Check if there's enough data for both groups
if len(on_time_reviews) >= 2 and len(late_reviews) >= 2:
    # Perform independent two-sample t-test
    t_stat, p_value = stats.ttest_ind(on_time_reviews, late_reviews, equal_var=False) # Use Welch's t-test (equal_var=False) as sample sizes and variances might differ

    print(f"\nMean review score for On Time deliveries: {on_time_reviews.mean():.4f}")
    print(f"Mean review score for Late deliveries: {late_reviews.mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpret the result
    alpha = 0.05
    print(f"\nSignificance level (alpha): {alpha}")

    if p_value < alpha:
        print("Conclusion: Reject the null hypothesis.")
        print("There is a statistically significant difference in the average review score between on-time and late deliveries.")
    else:
        print("Conclusion: Fail to reject the null hypothesis.")
        print("There is no statistically significant difference in the average review score between on-time and late deliveries.")

else:
    print("\nNot enough data for both On Time and Late delivery groups to perform t-test.")


# 2. Payment Type vs. High Value Orders

- Null Hypothesis (H0): Payment type group and value group are independent. (No association)
- Alternative Hypothesis (H1): Payment type group and value group are dependent. (There is an association)

multi_payment_and_value_data_query = """
SELECT
    o.order_id,
    CASE WHEN op.distinct_payment_types > 1 THEN 'Multiple Types' ELSE 'Single Type' END AS payment_type_group,
    CASE WHEN op.total_value >1000 THEN 'Value > 1000' ELSE 'Value <= 1000' END AS value_group
FROM
    orders o
JOIN (
    SELECT order_id, COUNT(DISTINCT payment_type) as distinct_payment_types, SUM(payment_value) as total_value
    FROM order_payments
    GROUP BY order_id
) op ON o.order_id = op.order_id
WHERE o.order_delivered_customer_date IS NOT NULL;
"""
multi_payment_and_value_data_df = pd.read_sql_query(multi_payment_and_value_data_query, conn)

if not multi_payment_and_value_data_df.empty:
    # Create a contingency table
    contingency_table = pd.crosstab(
        multi_payment_and_value_data_df['payment_type_group'],
        multi_payment_and_value_data_df['value_group']
    )

    print("\nContingency Table:")
    display(contingency_table)

    if contingency_table.shape == (2, 2) and contingency_table.sum().sum() > 0 and (contingency_table > 5).all().all(): # Check if it's 2x2 and cell counts are large enough (rule of thumb)
        # Perform Chi-Squared Test of Independence
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        print(f"\nChi-squared statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        print("Expected frequencies:\n", expected)

        # Interpret the result
        alpha = 0.05
        print(f"\nSignificance level (alpha): {alpha}")

        if p_value < alpha:
            print("Conclusion: Reject the null hypothesis.")
            print("There is a statistically significant association between using multiple payment types and the order having a total value greater than $1000.")
        else:
            print("Conclusion: Fail to reject the null hypothesis.")
            print("There is no statistically significant association between using multiple payment types and the order having a total value greater than $1000.")
    else:
        print("\nContingency table does not meet the criteria for Chi-Squared test (e.g., not 2x2, or cell counts too low).")
        print("Consider using Fisher's Exact Test for small counts or check data structure.")

else:
    print("\nNot enough data available to create a contingency table for the Chi-Squared test.")


# **Insights from Hypothesis Testing: Payment Type vs. High Value Orders**


 Chi-Squared test statistically evaluates whether there is a relationship between how customers pay (single vs. multiple methods) and whether their order is high-value (>\$1000). A statistically significant result validates the visual and descriptive observation that high-value orders are more likely to involve multiple payment types, reinforcing the importance of offering payment flexibility for large purchases.

# 3. Average Profit Margin
## Is the average profit margin for the 'computers' product category significantly different from a hypothesized value, say $50?

- Null Hypothesis (H0): The average profit margin for 'computers' is equal to $50 (μ = 50)

- Alternative Hypothesis (H1): The average profit margin for 'computers' is different from $50 (μ ≠ 50)

# Get profit margins for 'computers' category
computers_margin_query = """
SELECT
    (oi.price - oi.freight_value) AS profit_margin
FROM
    order_items oi
JOIN
    products p ON oi.product_id = p.product_id
LEFT JOIN
    product_cat_name pc ON p.product_category_name = pc.product_category_name
WHERE
    pc.product_category_name_english = 'computers'
    AND oi.price IS NOT NULL
    AND oi.freight_value IS NOT NULL;
"""
computers_margins = pd.read_sql_query(computers_margin_query, conn)['profit_margin'].dropna()

# Hypothesized mean
mu_0 = 50.0
print(f"\nHypothesized average profit margin for 'computers': ${mu_0}")

if len(computers_margins) >= 2:
    # Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(computers_margins, mu_0)

    print(f"\nSample mean profit margin for 'computers': {computers_margins.mean():.4f}")
    print(f"Sample size: {len(computers_margins)}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpret the result
    alpha = 0.05
    print(f"\nSignificance level (alpha): {alpha}")

    if p_value < alpha:
        print("Conclusion: Reject the null hypothesis.")
        print(f"The average profit margin for 'computers' is statistically significantly different from ${mu_0}.")
    else:
        print("Conclusion: Fail to reject the null hypothesis.")
        print(f"There is no statistically significant evidence to say the average profit margin for 'computers' is different from ${mu_0}.")

else:
    print("\nNot enough data for the 'computers' product category to perform one-sample t-test.")


**Insights are valuable for strategic decisions:**

1. High margin categories might offer flexibility for discounts or bundling.

2. The limitation is that this is a simplified profit margin calculation (Price - Freight Value). A true profit margin would account for COGS, operational costs, etc.

3. The hypothesis test specifically checked if the 'computers' category's average margin is statistically different from $50.

The conclusion of that test (based on the p-value) will determine if the observed sample mean for 'computers' is likely a true difference or just due to random chance.

# 4. Review Score vs. Payment Type

- Null Hypothesis (H0): The average review score is the same across all payment types (μ1 = μ2 = ... = μk)
- Alternative Hypothesis (H1): At least one payment type has a different average review score.

# Get review scores for each payment type
review_score_by_payment_type_query = """
SELECT
    op.payment_type,
    r.review_score
FROM
    order_payments op
JOIN
    orders o ON op.order_id = o.order_id -- Ensure delivered orders if needed, or use all orders with reviews
JOIN
    order_reviews r ON o.order_id = r.order_id
WHERE
    r.review_score IS NOT NULL
    AND op.payment_type IS NOT NULL
    AND o.order_status IN ('delivered', 'shipped'); -- Consider relevant order statuses
"""
review_score_by_payment_type_df = pd.read_sql_query(review_score_by_payment_type_query, conn)

# Get unique payment types
payment_types = review_score_by_payment_type_df['payment_type'].unique()

# Filter out payment types with insufficient data (less than 2 samples)
valid_payment_types = [pt for pt in payment_types if len(review_score_by_payment_type_df[review_score_by_payment_type_df['payment_type'] == pt]['review_score'].dropna()) >= 2]

if len(valid_payment_types) >= 2: # Need at least two groups for ANOVA
    # Prepare data for ANOVA: list of arrays, one for each group's review scores
    groups = [review_score_by_payment_type_df[review_score_by_payment_type_df['payment_type'] == pt]['review_score'].dropna().astype(float) for pt in valid_payment_types]

    # Perform one-way ANOVA

    f_stat, p_value = stats.f_oneway(*groups)

    print("\nAverage Review Score by Payment Type:")
    for pt in valid_payment_types:
        mean_score = review_score_by_payment_type_df[review_score_by_payment_type_df['payment_type'] == pt]['review_score'].mean()
        print(f"  {pt}: {mean_score:.4f}")


    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpret the result
    alpha = 0.05
    print(f"\nSignificance level (alpha): {alpha}")

    if p_value < alpha:
        print("Conclusion: Reject the null hypothesis.")
        print("There is a statistically significant difference in the average review score among different payment types.")
    else:
        print("Conclusion: Fail to reject the null hypothesis.")
        print("There is no statistically significant evidence to say the average review score is different among different payment types.")

else:
    print("\nNot enough payment types with sufficient data (at least 2 samples) to perform ANOVA.")


# 5. Price vs. Freight Value
## Is there a significant linear relationship between the price of an item and its freight value?

# Get price and freight value from order_items
price_freight_df = order_items[['price', 'freight_value']].dropna()

if len(price_freight_df) >= 2:
    # Perform Pearson correlation test

    correlation_coefficient, p_value = stats.pearsonr(price_freight_df['price'], price_freight_df['freight_value'])

    print(f"\nPearson correlation coefficient between price and freight value: {correlation_coefficient:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpret the result
    alpha = 0.05
    print(f"\nSignificance level (alpha): {alpha}")

    if p_value < alpha:
        print("Conclusion: Reject the null hypothesis.")
        print("There is a statistically significant linear relationship between the price of an item and its freight value.")
        print(f"The correlation coefficient ({correlation_coefficient:.4f}) suggests the nature and strength of this relationship.")
    else:
        print("Conclusion: Fail to reject the null hypothesis.")
        print("There is no statistically significant evidence to say there is a linear relationship between the price of an item and its freight value.")

else:
    print("\nNot enough data with both price and freight value available to perform Pearson correlation test.")

# A/B Testing

## Scenario 1: Impact of Free Shipping Threshold

### Does introducing a free shipping threshold of $200 significantly increase the average order value compared to the previous shipping policy (variable freight based on price)?

- Null Hypothesis (H0): The average order value with the $200 free shipping threshold (Treatment Group) is the same as the average order value with the old policy (Control Group). μ_treatment = μ_control

- Alternative Hypothesis (H1): The average order value with the $200 free shipping threshold is significantly higher than the average order value with the old policy. μ_treatment > μ_control


# Simulate A/B test group assignment for orders
# In a real A/B test, this assignment would happen when a user interacts with the tested feature.
# Here, we'll randomly assign existing orders to groups for demonstration purposes.

# Get all unique orders
all_orders = orders[['order_id', 'customer_id']].copy()

# Simulate random assignment to Control (old policy) or Treatment (new free shipping) group
# We'll assign approximately 50% to each group
all_orders['ab_group'] = np.random.choice(['Control', 'Treatment'], size=len(all_orders), p=[0.5, 0.5])

# Calculate total value for each order by summing up payment values
order_total_values_query = """
SELECT
    order_id,
    SUM(payment_value) AS total_order_value
FROM
    order_payments
GROUP BY
    order_id;
"""
order_total_values_df = pd.read_sql_query(order_total_values_query, conn)

# Merge the simulated A/B group assignment with the order total values
ab_test_data = all_orders.merge(order_total_values_df, on='order_id', how='inner')

# Separate data for Control and Treatment groups
control_data_df = ab_test_data[ab_test_data['ab_group'] == 'Control'].copy()
treatment_data_df = ab_test_data[ab_test_data['ab_group'] == 'Treatment'].copy()

print("Simulated A/B Test Data Head:")
display(ab_test_data.head())

print("\nControl Group Data Head:")
display(control_data_df.head())

print("\nTreatment Group Data Head:")
display(treatment_data_df.head())

print(f"\nControl Group Size: {len(control_data_df)}")
print(f"Treatment Group Size: {len(treatment_data_df)}")

# Now, we can use these simulated dataframes to perform the t-test as outlined before.
# We will rename 'total_order_value' to 'order_value' to match the previous code's expectation.
control_data_df.rename(columns={'total_order_value': 'order_value'}, inplace=True)
treatment_data_df.rename(columns={'total_order_value': 'order_value'}, inplace=True)

# Now the previous t-test code should work with these simulated dataframes.

# Perform the independent two-sample t-test on the simulated data

if 'order_value' in control_data_df.columns and 'order_value' in treatment_data_df.columns:
     control_aov = control_data_df['order_value'].dropna()
     treatment_aov = treatment_data_df['order_value'].dropna()

     if len(control_aov) >= 2 and len(treatment_aov) >= 2:
         # Perform one-sided t-test (alternative='greater' if testing for increase)
         t_stat, p_value = stats.ttest_ind(treatment_aov, control_aov, equal_var=False, alternative='greater')

         print("\n--- A/B Test Analysis (Scenario 1) ---")
         print(f"Control Group AOV: {control_aov.mean():.2f} (n={len(control_aov)})")
         print(f"Treatment Group AOV: {treatment_aov.mean():.2f} (n={len(treatment_aov)})")
         print(f"T-statistic: {t_stat:.4f}")
         print(f"P-value: {p_value:.4f}")

         alpha = 0.05
         print(f"\nSignificance level (alpha): {alpha}")

         if p_value < alpha:
             print("Conclusion: Reject H0.")
             print("There is a statistically significant increase in Average Order Value (AOV) in the Treatment group compared to the Control group.")
             print("This suggests that the simulated free shipping threshold had a positive impact on AOV.")
         else:
             print("Conclusion: Fail to reject H0.")
             print("There is no statistically significant evidence to suggest an increase in Average Order Value (AOV) in the Treatment group compared to the Control group.")
             print("This suggests that the simulated free shipping threshold did not have a significant impact on AOV.")

     else:
         print("\nNot enough data in A/B test groups after dropping NaNs for t-test.")
else:
    print("\n'order_value' column not found in one or both simulated A/B test dataframes.")

## Scenario 2: Impact of Targeted Promotions on Inactive Customers
###  Does a targeted re-engagement email campaign sent specifically to inactive customers in Sao Paulo (SP) result in a significantly higher conversion rate (placing an order) compared to a generic re-engagement email campaign sent to inactive customers in other states?

- Null Hypothesis (H0): The conversion rate for inactive SP customers receiving the targeted campaign (Treatment Group) is the same as the conversion rate for inactive customers in other states receiving the generic campaign (Control Group). p_treatment = p_control
- Alternative Hypothesis (H1): The conversion rate for inactive SP customers receiving the targeted campaign is significantly higher than the conversion rate for inactive customers in other states receiving the generic campaign. p_treatment > p_control


# Scenario 2: Impact of Targeted Promotions on Inactive Customers
# Simulate Control and Treatment Groups for the targeted re-engagement campaign

# We already have the inactive customers identified in `inactive_customers_df`
# Separate inactive customers into Treatment (SP) and Control (Other States) groups

treatment_group_customers = inactive_customers_df[inactive_customers_df['customer_state'] == 'SP'].copy()
control_group_customers = inactive_customers_df[inactive_customers_df['customer_state'] != 'SP'].copy()

print("Treatment Group (Inactive Customers from SP) Head:")
display(treatment_group_customers.head())
print(f"Treatment Group Size: {len(treatment_group_customers)}")

print("\nControl Group (Inactive Customers from Other States) Head:")
display(control_group_customers.head())
print(f"Control Group Size: {len(control_group_customers)}")

# For the purpose of demonstrating the proportion z-test, we need simulated conversion counts.
# In a real A/B test, these would be actual counts from your email campaign tracking.
# Let's assume a hypothetical conversion rate for each group.
# For example, let's assume a slightly higher hypothetical conversion rate for the targeted SP group.

hypothetical_conversion_rate_treatment = 0.01 # 1% hypothetical conversion rate for SP group
hypothetical_conversion_rate_control = 0.008 # 0.8% hypothetical conversion rate for Other States group

# Simulate the number of conversions based on these hypothetical rates
# Use round() to get integer counts, as conversions are whole numbers
simulated_conversions_treatment = round(len(treatment_group_customers) * hypothetical_conversion_rate_treatment)
simulated_conversions_control = round(len(control_group_customers) * hypothetical_conversion_rate_control)


# Create dictionaries to hold the summary statistics for the proportion z-test
control_ab_summary = {
    'conversions': simulated_conversions_control,
    'total_customers': len(control_group_customers)
}

treatment_ab_summary = {
    'conversions': simulated_conversions_treatment,
    'total_customers': len(treatment_group_customers)
}

print("\nSimulated Control Group Summary for Proportion Test:")
display(control_ab_summary)

print("\nSimulated Treatment Group Summary for Proportion Test:")
display(treatment_ab_summary)

# Now, the code for the proportion z-test (from the previous failed cell) should work with these summary dictionaries.

# Perform the independent two-sample proportion z-test on the simulated data

# Assuming you have counts of conversions and total customers for each group
if 'conversions' in control_ab_summary and 'total_customers' in control_ab_summary:
     count_control = control_ab_summary['conversions']
     nobs_control = control_ab_summary['total_customers']
     count_treatment = treatment_ab_summary['conversions']
     nobs_treatment = treatment_ab_summary['total_customers']

     if nobs_control > 0 and nobs_treatment > 0:
         # Perform one-sided proportion z-test (alternative='larger' if testing for increase)
         # Null Hypothesis (H0): The conversion rate for inactive SP customers receiving the targeted campaign (Treatment Group) is the same as the conversion rate for inactive customers in other states receiving the generic campaign (Control Group). p_treatment = p_control
         # Alternative Hypothesis (H1): The conversion rate for inactive SP customers receiving the targeted campaign is significantly higher than the conversion rate for inactive customers in other states receiving the generic campaign. p_treatment > p_control
         stat, p_value = proportions_ztest([count_treatment, count_control], [nobs_treatment, nobs_control], alternative='larger')

         print("\n--- A/B Test Analysis (Scenario 2) ---")
         print(f"Control Group Conversion Rate: {count_control/nobs_control:.4f} ({count_control}/{nobs_control})")
         print(f"Treatment Group Conversion Rate: {count_treatment/nobs_treatment:.4f} ({count_treatment}/{nobs_treatment})")
         print(f"Z-statistic: {stat:.4f}")
         print(f"P-value: {p_value:.4f}")

         alpha = 0.05
         print(f"\nSignificance level (alpha): {alpha}")

         if p_value < alpha:
             print("Conclusion: Reject H0.")
             print("There is a statistically significant increase in conversion rate in the Treatment group compared to the Control group.")
             print("This suggests that the simulated targeted campaign had a positive impact on conversion rate.")
         else:
             print("Conclusion: Fail to reject H0.")
             print("There is no statistically significant evidence to suggest an increase in conversion rate in the Treatment group compared to the Control group.")
             print("This suggests that the simulated targeted campaign did not have a significant impact on conversion rate.")
     else:
         print("\nNot enough data in A/B test groups for proportion z-test.")
else:
    print("\n'conversions' or 'total_customers' not found in simulated A/B test summary dictionaries.")

#Business Recommendations

###1. Enhance Customer Lifetime Value (CLV):
   - Action: Focus marketing and customer retention efforts on high-CLV states and cities (identified in CLV analysis).
   - Recommendation: Implement loyalty programs, personalized offers, or exclusive deals for customers in these valuable geographic segments. Consider running targeted campaigns based on past purchase behavior within these areas.

###2. Address Customer Inactivity/Churn:
   - Action: Develop and execute targeted re-engagement campaigns for inactive customers, especially in high-inactivity states and cities (like SP).
   - Recommendation: Use insights from their past purchases (if any) or common purchase patterns of active customers in their region to personalize email or notification content. Offer incentives like discounts or free shipping to encourage a return purchase.

###3. Improve Delivery Performance:
  - Action: Prioritize optimizing logistics and seller performance to minimize delivery delays, as late deliveries significantly impact review scores.
  - Recommendation: Work closely with sellers identified as having high average delays (from Delivery Time Deviation analysis). Provide support, training, or stricter performance requirements. Consider improving estimated delivery time accuracy and proactively communicating potential delays to customers.

###4.Facilitate Multi-Payment Options for High-Value Orders:
    - Action: Ensure a seamless experience for customers using multiple payment types, as this is correlated with higher order values.
    - Recommendation: Review the checkout process to make combining payment methods easy and intuitive. Consider highlighting the availability of multiple payment options, especially for larger carts.

###5.Implement Product Recommendation Systems:
    - Action: Utilize the identified frequent product category co-occurrences  ("Customers Who Bought X Also Bought Y" analysis) to build or improve recommendation engines.
    - Recommendation: Add "Frequently Bought Together" sections on product pages based on the top co-occurring categories. Personalize recommendations based on a customer's browsing and purchase history, leveraging cross-category insights.

### 6. Manage Inventory Strategically:
    - Action: Ensure adequate stock levels for products within the top-selling categories (by revenue and order count) and categories with high average profit margins.
    - Recommendation: Use sales forecasts based on historical data and trends for these key categories. Collaborate with sellers or suppliers to manage inventory effectively and avoid stockouts for popular or high-margin items.

### 7. Enhance Seller Performance Management:
    - Action: Use the Seller Performance Index to identify both top-performing sellers (for recognition or best practice sharing) and underperforming sellers (for intervention).
   - Recommendation: Implement performance improvement plans for sellers with low composite scores, focusing on areas like delivery speed, review management, or order cancellation rates. Reward top sellers to encourage continued high performance.

### 8. Focus on Customer Retention:
    - Action: Given the likely low repeat purchase rate within 90 days, dedicate resources to converting first-time buyers into repeat customers.
    - Recommendation: Implement a post-purchase follow-up strategy. This could include thank-you emails, requests for reviews (which can provide valuable feedback), recommendations for related products, and targeted discounts for a second purchase. Analyze the time taken for repeat purchases to refine the timing of these campaigns.

### 9. Identify and Nurture Cross-Category Buyers:
    - Action: Recognize customers who purchase from three or more distinct categories as valuable segments.
    - Recommendation: Tailor marketing communications to highlight the breadth of available products across different categories. Offer curated collections or bundles that span multiple interests based on their past behavior. These customers may be more receptive to exploring new product areas on the platform.

### 10. Optimize High-Margin Categories:
     - Action: Analyze the categories with the highest average profit margins beyond just price minus freight.
     - Recommendation: Conduct deeper cost analysis for these categories (including COGS, marketing, etc.) to understand true profitability. Explore strategies to increase sales volume within these categories through targeted marketing, improved product listings, or competitive pricing reviews while maintaining healthy margins.

### 11. Address Low Review Scores for Specific Payment Types (if significant):
   - Action: If the ANOVA test indicates a statistically significant difference in review scores by payment type, investigate the specific payment methods associated with lower scores.
   - Recommendation: Examine the user experience for those payment types. Are there technical issues? Are instructions clear? Are there hidden fees? Addressing friction points in the payment process can improve overall customer satisfaction and review scores.

### 12. Investigate Price-Freight Relationship (if significant):
    - Action: If a significant linear relationship between price and freight value is found, investigate the underlying logistics or pricing models.
    - Recommendation: Understand why more expensive items might have higher freight costs. Is this due to size/weight, value-based shipping insurance, or seller-specific practices? Optimize freight calculation methods to ensure competitiveness while covering costs.

    




#Business Recommendations

###1. Enhance Customer Lifetime Value (CLV):
   - Action: Focus marketing and customer retention efforts on high-CLV states and cities (identified in CLV analysis).
   - Recommendation: Implement loyalty programs, personalized offers, or exclusive deals for customers in these valuable geographic segments. Consider running targeted campaigns based on past purchase behavior within these areas.

###2. Address Customer Inactivity/Churn:
   - Action: Develop and execute targeted re-engagement campaigns for inactive customers, especially in high-inactivity states and cities (like SP).
   - Recommendation: Use insights from their past purchases (if any) or common purchase patterns of active customers in their region to personalize email or notification content. Offer incentives like discounts or free shipping to encourage a return purchase.

###3. Improve Delivery Performance:
  - Action: Prioritize optimizing logistics and seller performance to minimize delivery delays, as late deliveries significantly impact review scores.
  - Recommendation: Work closely with sellers identified as having high average delays (from Delivery Time Deviation analysis). Provide support, training, or stricter performance requirements. Consider improving estimated delivery time accuracy and proactively communicating potential delays to customers.

###4.Facilitate Multi-Payment Options for High-Value Orders:
- **Action:** Ensure a seamless experience for customers using multiple payment types, as this is correlated with higher order values.

- **Recommendation:** Review the checkout process to make combining payment methods easy and intuitive. Consider highlighting the available options.

###5.Implement Product Recommendation Systems:
- Action: Utilize the identified frequent product category co-occurrences  ("Customers Who Bought X Also Bought Y" analysis) to build or improve recommendation engines.
- Recommendation: Add "Frequently Bought Together" sections on product pages based on the top co-occurring categories. Personalize recommendations based on a customer's browsing and purchase history, leveraging cross-category insights.

### 6. Manage Inventory Strategically:
- Action: Ensure adequate stock levels for products within the top-selling categories (by revenue and order count) and categories with high average profit margins.
- Recommendation: Use sales forecasts based on historical data and trends for these key categories. Collaborate with sellers or suppliers to manage inventory effectively and avoid stockouts for popular or high-margin items.

### 7. Enhance Seller Performance Management:
- Action: Use the Seller Performance Index to identify both top-performing sellers (for recognition or best practice sharing) and underperforming sellers (for intervention).
- Recommendation: Implement performance improvement plans for sellers with low composite scores, focusing on areas like delivery speed, review management, or order cancellation rates. Reward top sellers to encourage continued high performance.

### 8. Focus on Customer Retention:
- Action: Given the likely low repeat purchase rate within 90 days, dedicate resources to converting first-time buyers into repeat customers.
- Recommendation: Implement a post-purchase follow-up strategy. This could include thank-you emails, requests for reviews (which can provide valuable feedback), recommendations for related products, and targeted discounts for a second purchase. Analyze the time taken for repeat purchases to refine the timing of these campaigns.

### 9. Identify and Nurture Cross-Category Buyers:
- Action: Recognize customers who purchase from three or more distinct categories as valuable segments.
- Recommendation: Tailor marketing communications to highlight the breadth of available products across different categories. Offer curated collections or bundles that span multiple interests based on their past behavior. These customers may be more receptive to exploring new product areas on the platform.

### 10. Optimize High-Margin Categories:
- Action: Analyze the categories with the highest average profit margins beyond just price minus freight.
- Recommendation: Conduct deeper cost analysis for these categories (including COGS, marketing, etc.) to understand true profitability. Explore strategies to increase sales volume within these categories through targeted marketing, improved product listings, or competitive pricing reviews while maintaining healthy margins.

### 11. Address Low Review Scores for Specific Payment Types (if significant):
- Action: If the ANOVA test indicates a statistically significant difference in review scores by payment type, investigate the specific payment methods associated with lower scores.
- Recommendation: Examine the user experience for those payment types. Are there technical issues? Are instructions clear? Are there hidden fees? Addressing friction points in the payment process can improve overall customer satisfaction and review scores.

### 12. Investigate Price-Freight Relationship (if significant):
- Action: If a significant linear relationship between price and freight value is found, investigate the underlying logistics or pricing models.
- Recommendation: Understand why more expensive items might have higher freight costs. Is this due to size/weight, value-based shipping insurance, or seller-specific practices? Optimize freight calculation methods to ensure competitiveness while covering costs.

    






