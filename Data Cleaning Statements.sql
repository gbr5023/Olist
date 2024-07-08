/* Created tables for reference
select * from customer limit 10;
select * from geolocation limit 10;
select * from orders limit 10;
select * from order_item limit 10;
select * from product_category limit 10;
select * from product limit 10;
select * from seller limit 10;
select * from review limit 10;
select * from geolocation_state;
*/

/*
Null Values
*/
-- Customer
select 
	(case when (sum(case when cust_id is null then 1 else 0 end)) > 0 then True else False end) as is_custid_null,
	(case when (sum(case when cust_unique_id is null then 1 else 0 end)) > 0 then True else False end) as is_custuniqueid_null,
	(case when (sum(case when cust_zip_cd_prefix is null then 1 else 0 end)) > 0 then True else False end) as is_zip_null,
	(case when (sum(case when cust_city is null then 1 else 0 end)) > 0 then True else False end) as is_city_null,
	(case when (sum(case when cust_state is null then 1 else 0 end)) > 0 then True else False end) as is_state_null
from customer; 
-- Customer has no NULL values
------------------------------------------------------------------------------------------------------------------------

-- Geolocation
select
	(case when (sum(case when zip_cd_prefix is null then 1 else 0 end)) > 0 then True else False end) as is_zip_null,
	(case when (sum(case when geoloc_lat is null then 1 else 0 end)) > 0 then True else False end) as is_lat_null,
	(case when (sum(case when geoloc_long is null then 1 else 0 end)) > 0 then True else False end) as is_long_null,
	(case when (sum(case when geoloc_city is null then 1 else 0 end)) > 0 then True else False end) as is_city_null,
	(case when (sum(case when geoloc_state is null then 1 else 0 end)) > 0 then True else False end) as is_state_null
from geolocation;
-- Geolocation has no NULL values
------------------------------------------------------------------------------------------------------------------------

-- Geolocation_State
select
	(case when (sum(case when geoloc_state is null then 1 else 0 end)) > 0 then True else False end) as is_state_null,
	(case when (sum(case when geoloc_state_name is null then 1 else 0 end)) > 0 then True else False end) as is_name_null,
	(case when (sum(case when geoloc_region is null then 1 else 0 end)) > 0 then True else False end) as is_region_null,
	(case when (sum(case when geoloc_state_pop is null then 1 else 0 end)) > 0 then True else False end) as is_pop_null
from geolocation_state;
-- Geolocation_State has no NULL values
------------------------------------------------------------------------------------------------------------------------

-- Orders
select
	(case when (sum(case when order_id is null then 1 else 0 end)) > 0 then True else False end) as is_orderid_null,
	(case when (sum(case when cust_id is null then 1 else 0 end)) > 0 then True else False end) as is_custidd_null,
	(case when (sum(case when order_status is null then 1 else 0 end)) > 0 then True else False end) as is_status_null,
	(case when (sum(case when order_purchase_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_purchasedtm_null,
	(case when (sum(case when order_approved_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_approveddtm_null,
	(case when (sum(case when order_carrier_delivery_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_carrierdelivery_null,
	(case when (sum(case when order_cust_delivery_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_custdelivery_null,
	(case when (sum(case when order_estm_delivery_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_estmdelivery_null
from orders;
-- Orders has NULL values in the order_approved_dtm, order_carrier_delivery_dtm, & order_cust_delivery_dtm fields
-- Remove unnecessary fields for case study
alter table orders
	drop column order_approved_dtm,
	drop column order_carrier_delivery_dtm;

select
	(case when (sum(case when order_id is null then 1 else 0 end)) > 0 then True else False end) as is_orderid_null,
	(case when (sum(case when cust_id is null then 1 else 0 end)) > 0 then True else False end) as is_custidd_null,
	(case when (sum(case when order_status is null then 1 else 0 end)) > 0 then True else False end) as is_status_null,
	(case when (sum(case when order_purchase_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_purchasedtm_null,
	(case when (sum(case when order_cust_delivery_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_custdelivery_null,
	(case when (sum(case when order_estm_delivery_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_estmdelivery_null
from orders;

-- Remove rows where order_cust_delivery_dtm is NULL and order_status is not 'delivered'
select count(*)
from orders
where order_cust_delivery_dtm is NULL; -- 2,965 orders to delete

delete from orders
where order_cust_delivery_dtm is NULL;
delete from orders
where order_status != 'delivered';

-- add order date column for convenience
alter table orders
add column order_dt date;

update orders
set order_dt = order_purchase_dtm::date;

select * from orders;

select order_status, count(distinct order_id)
from orders
group by 1; -- orders: 96,470 unique orders

------------------------------------------------------------------------------------------------------------------------
/*
Need to clean Customer table to remove customers whose orders were removed in the Orders clean up
*/
select *
from customer
where cust_id not in (select cust_id from orders); -- 2,971 customers to delete

delete from customer
where cust_id not in (select cust_id from orders);

select count(distinct cust_id) from customer; -- customer: 96,470 unique customers
------------------------------------------------------------------------------------------------------------------------

-- Order_Item
select
	(case when (sum(case when order_id is null then 1 else 0 end)) > 0 then True else False end) as is_orderId_null,
	(case when (sum(case when order_item_id is null then 1 else 0 end)) > 0 then True else False end) as is_itemId_null,
	(case when (sum(case when product_id is null then 1 else 0 end)) > 0 then True else False end) as is_productId_null,
	(case when (sum(case when seller_id is null then 1 else 0 end)) > 0 then True else False end) as is_sellerId_null,
	(case when (sum(case when shipping_limit_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_shippingLimitDtm_null,
	(case when (sum(case when product_price is null then 1 else 0 end)) > 0 then True else False end) as is_price_null,
	(case when (sum(case when product_ship_fee is null then 1 else 0 end)) > 0 then True else False end) as is_shipFee_null
from order_item;

/*
Need to clean Order_Item table for orders that were removed in the Orders clean up
*/
select *
from order_item
where order_id not in (select order_id from orders); -- 2,461 orders to delete

delete from order_item
where order_id not in (select order_id from orders);

select count(distinct order_id) from order_item; -- customer: 96,470 unique orders
------------------------------------------------------------------------------------------------------------------------

-- Product_Category
select
	(case when (sum(case when category_name is null then 1 else 0 end)) > 0 then True else False end) as is_name_null,
	(case when (sum(case when category_name_en is null then 1 else 0 end)) > 0 then True else False end) as is_nameEn_null
from product_category;
-- Product_Category has no NULL values

-- check for missing values in product_category
select category_name
from product
where category_name not in (select category_name from product_category)
group by 1;
/* missing
"portateis_cozinha_e_preparadores_de_alimentos"
"pc_gamer"
*/

-- Insert missing values into product_category reference table
insert into product_category (
	category_name,
	category_name_en)
values
	('portateis_cozinha_e_preparadores_de_alimentos', 'portable_kitchen_food_processors'),
	('pc_gamer', 'pc_gamer');

select category_name
from product
where category_name not in (select category_name from product_category)
group by 1;
------------------------------------------------------------------------------------------------------------------------

-- Product
select
	(case when (sum(case when product_id is null then 1 else 0 end)) > 0 then True else False end) as is_productId_null,
	(case when (sum(case when category_name is null then 1 else 0 end)) > 0 then True else False end) as is_categoryName_null,
	(case when (sum(case when name_length is null then 1 else 0 end)) > 0 then True else False end) as is_nameLength_null,
	(case when (sum(case when description_length is null then 1 else 0 end)) > 0 then True else False end) as is_descLength_null,
	(case when (sum(case when photos_qty is null then 1 else 0 end)) > 0 then True else False end) as is_photosQty_null,
	(case when (sum(case when weight_g is null then 1 else 0 end)) > 0 then True else False end) as is_weightG_null,
	(case when (sum(case when length_cm is null then 1 else 0 end)) > 0 then True else False end) as is_lengthCm_null,
	(case when (sum(case when height_cm is null then 1 else 0 end)) > 0 then True else False end) as is_heightCm_null,
	(case when (sum(case when width_cm is null then 1 else 0 end)) > 0 then True else False end) as is_widthCm_null
from product;

-- Remove unnecessary fields
alter table product
	drop column name_length,
	drop column description_length,
	drop column photos_qty,
	drop column weight_g,
	drop column length_cm,
	drop column height_cm,
	drop column width_cm;

select
	(sum(case when product_id is null then 1 else 0 end)) as is_productId_null,
	(sum(case when category_name is null then 1 else 0 end)) as is_categoryName_null
from product;

-- Populate NULL values with "Not Defined"
update product
	set category_name = 'Not Defined'
where category_name is null;

-- insert Not Defined product category in product_category table
insert into product_category(
	category_name,
	category_name_en
)
values 
	('Not Defined', 'Not Defined');

-- Remove products not included in any orders
select product_id
from product 
where product_id not in (select product_id from order_item); -- 737 products to remove

delete from product
where product_id not in (select product_id from order_item);

select count(distinct product_id) from product; -- 32,214 unique products
------------------------------------------------------------------------------------------------------------------------

-- Seller
select
	(case when (sum(case when seller_id is null then 1 else 0 end)) > 0 then True else False end) as is_sellerId_null,
	(case when (sum(case when seller_zip_cd_prefix is null then 1 else 0 end)) > 0 then True else False end) as is_zip_null,
	(case when (sum(case when seller_city is null then 1 else 0 end)) > 0 then True else False end) as is_city_null,
	(case when (sum(case when seller_state is null then 1 else 0 end)) > 0 then True else False end) as is_state_null
from seller;
-- Seller has no NULL values

select count(distinct seller_id) from seller;

-- Remove sellers with no associated orders
select seller_id
from seller
where seller_id not in (select seller_id from order_item); -- 125 sellers to remove

delete from seller
where seller_id not in (select seller_id from order_item);

select count(distinct seller_id)
from seller; -- 2,970 unique sellers

------------------------------------------------------------------------------------------------------------------------

-- Review
select
	(case when (sum(case when review_id is null then 1 else 0 end)) > 0 then True else False end) as is_reviewId_null,
	(case when (sum(case when order_id is null then 1 else 0 end)) > 0 then True else False end) as is_orderId_null,
	(case when (sum(case when review_score is null then 1 else 0 end)) > 0 then True else False end) as is_score_null,
	(case when (sum(case when review_title is null then 1 else 0 end)) > 0 then True else False end) as is_title_null,
	(case when (sum(case when review_message is null then 1 else 0 end)) > 0 then True else False end) as is_message_null,
	(case when (sum(case when review_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_reviewDtm_null,
	(case when (sum(case when review_response_dtm is null then 1 else 0 end)) > 0 then True else False end) as is_reviewResponseDtm_null
from review;

-- Remove orders from review table that were removed in the Order table cleanup
select count(*)
from review
where order_id not in (select order_id from orders); -- 2,871 orders to remove

delete from review
where order_id not in (select order_id from orders);

-- title and message have NULL, how many?
select
	(sum(case when review_title is null then 1 else 0 end)) as is_title_null,
	(sum(case when review_message is null then 1 else 0 end)) as is_message_null
from review;
/* 85149	
57259 */
-- How many unique reviews and orders?
select 
	count(distinct review_id) as unique_reviews ,
	count(distinct order_id) as unique_orders
from review;

-- Populate NULL values with "Not Defined"
update review
	set review_title = 'Not Defined'
where review_title is null;
update review
	set review_message = 'Not Defined'
where review_message is null;
------------------------------------------------------------------------------------------------------------------------

/*
Translation Corrections
*/
select *
from product_category
order by category_name_en asc
;

update product_category
set category_name_en = 'cinema_picture'
where category_name_en = 'cine_photo';

update product_category
set category_name_en = 'construction_tools_garden'
where category_name_en = 'costruction_tools_garden';

update product_category
set category_name_en = 'construction_tools_tools'
where category_name_en = 'costruction_tools_tools';

update product_category
set category_name_en = 'fashion_underwear_beachwear'
where category_name_en = 'fashion_underwear_beach';

update product_category
set category_name_en = 'fashion_female_clothing'
where category_name_en = 'fashio_female_clothing';

update product_category
set category_name_en = 'landline_telephone'
where category_name_en = 'fixed_telephony';

update product_category
set category_name_en = 'console_games'
where category_name_en = 'consoles_games';

update product_category
set category_name_en = 'baby_diapers_and_hygiene'
where category_name_en = 'diapers_and_hygiene';

update product_category
set category_name_en = 'dvds_bluray'
where category_name_en = 'dvds_blu_ray';

update product_category
set category_name_en = 'fashion_children_clothing'
where category_name_en = 'fashion_childrens_clothes';

update product_category
set category_name_en = 'home_appliances_other'
where category_name_en = 'home_appliances_2';

update product_category
set category_name_en = 'guest_house_other'
where category_name_en = 'home_comfort_2';

update product_category
set category_name_en = 'guest_house'
where category_name_en = 'home_confort';

update product_category
set category_name_en = 'kitchenware_cookware'
where category_name_en = 'la_cuisine';

update product_category
set category_name_en = 'marketplace'
where category_name_en = 'market_place';

update product_category
set category_name_en = 'appliances_kitchen_food_processors'
where category_name_en = 'portable_kitchen_food_processors';

update product_category
set category_name_en = 'insurance_and_services'
where category_name_en = 'security_and_services';

update product_category
set category_name_en = 'safety_and_signage'
where category_name_en = 'signaling_and_security';

update product_category
set category_name_en = 'appliances_electronic'
where category_name_en = 'small_appliances';

update product_category
set category_name_en = 'appliances_home_oven_coffee'
where category_name_en = 'small_appliances_home_oven_and_coffee';

update product_category
set category_name_en = 'tablet_image_printing'
where category_name_en = 'tablets_printing_image';

update product_category
set category_name_en = 'telephone '
where category_name_en = 'telephony';

update product_category
set category_name_en = 'arts_and_crafts'
where category_name_en = 'arts_and_craftmanship';

update product_category
set category_name_en = 'construction_tools'
where category_name_en = 'construction_tools_tools';

