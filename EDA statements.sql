/*
Quick Analyses
*/

select oi.*,
	pr.category_name,
	pc.category_name_en
from order_item oi
left join product pr
	on oi.product_id = pr.product_id
left join product_category pc
	on pr.category_name = pc.category_name;

select count(distinct order_id) as orders_cnt
from order_item;
-- 96,470 unique orders

select seller_id,
	count(distinct order_id)
from order_item
group by 1
order by 2 desc;

select count(distinct order_id), count(distinct product_id), count(distinct seller_id)
from order_item;

/*
Analyses benefitting Olist (company)
- Speaking to the time frame, how many orders have been completed by Olist sellers?
- What geographical markets have the most orders? Are there trends for more orders in urban areas vs rural areas?
- What are the most popular product categories? most lucrative product categories? Is there an overlap?
- Do Olist sellers specialize in specific product categories or a variety?
- Are there sellers that dominate specific product categories (market share)?
- Which sellers are making the most money?
-
*/

-- Speaking to the time frame, how many orders have been completed by Olist sellers?
select extract(year from order_dt),
	extract(month from order_dt),
	count(distinct order_id)
from orders
group by 1,2
order by 1,2;

select extract(year from order_dt),
--	extract(month from order_dt),
	count(distinct order_id)
from orders
group by 1
order by 1;
/*
Orders date range is from September 2016 to August 2018 
In 2016, Olist began reaching out to sellers after receiving funding (explanation for slow start/low volumes in 2016)
*/
select cust_state
from customer
group by cust_state;

--  What geographical markets have the most orders? Are there trends for more orders in urban areas vs rural areas?
select 
	cust.cust_state,
	gs.geoloc_state_name,
	count(distinct o.order_id) as orders_cnt
from orders o
inner join customer cust
	on o.cust_id = cust.cust_id
inner join geolocation_state gs
	on cust.cust_state = gs.geoloc_state
group by 1,2
order by 3 desc;

select 
	gs.geoloc_region,
	count(distinct o.order_id) as orders_cnt
from orders o
inner join customer cust
	on o.cust_id = cust.cust_id
inner join geolocation_state gs
	on cust.cust_state = gs.geoloc_state
group by 1
order by 2 desc;

-- What are the most popular product categories? most lucrative product categories? Is there an overlap?
- Do Olist sellers specialize in specific product categories or a variety?
- Are there sellers that dominate specific product categories (market share)?
- Which sellers are making the most money?