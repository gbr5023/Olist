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
select 
	pc.category_name,
	pc.category_name_en,
	count(distinct oi.order_id),
	sum(oi.product_price)
from order_item oi
inner join product prod
	on oi.product_id = prod.product_id
inner join product_category pc
	on prod.category_name = pc.category_name
group by 1,2
order by 4 desc; --swap out 3 for 4 to check popular vs lucrative

select 
pc.category_name_en,
rank() over(order by count (distinct oi.order_id) desc) as pop_ranking,
rank() over(order by sum(oi.product_price) desc) as rev_ranks
--	count(distinct oi.order_id) as order_cnt
from order_item oi
inner join product prod
	on oi.product_id = prod.product_id
inner join product_category pc
	on prod.category_name = pc.category_name
group by 1
limit 10
;

	
/*
Most Popular
"bed_bath_table"	9272
"health_beauty"	8647
"sports_leisure"	7529
"computers_accessories"	6529
"furniture_decor"	6307
"housewares"	5743
"watches_gifts"	5493
"telephone "	4093
"auto"	3809
"toys"	3803

Most Lucrative
"health_beauty"		1233131.72
"watches_gifts"		1165898.98
"bed_bath_table"	1023434.76
"sports_leisure"	954673.55
"computers_accessories"		888613.62
"furniture_decor"	711927.69
"housewares"	615628.69
"cool_stuff"	610204.10
"auto"	578849.35
"toys"	471097.49

Save for telephone products/services there is overlap of the top 10 categories sold/offered by Olist sellers
"bed_bath_table"		1	3
"health_beauty"			2	1
"sports_leisure"		3	4
"computers_accessories"	4	5
"furniture_decor"		5	6
"housewares"			6	7
"watches_gifts"			7	2
"telephone "			8	14
"auto"					9	9
"toys"					10	10
*/

-- Do Olist sellers specialize in specific product categories or a variety?
select oi.seller_id,
	count(distinct prod.category_name),
	count(distinct oi.product_id)
from order_item oi
inner join product prod
	on oi.product_id = prod.product_id
inner join product_category prodcat
	on prod.category_name = prodcat.category_name
group by 1
order by 3 desc;

select prodcat.category_name_en,
	count(distinct oi.seller_id)
from product_category prodcat
inner join product prod
	on prodcat.category_name = prod.category_name
inner join order_item oi
	on prod.product_id = oi.product_id
group by 1
order by 2 desc;


-- Are there sellers that dominate specific product categories (market share)?
-- Which sellers are making the most money?