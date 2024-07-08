CREATE TABLE Customer (
  cust_id varchar(50),
  cust_unique_id varchar(50),
  cust_zip_cd_prefix varchar(5),
  cust_city varchar(50),
  cust_state varchar(2)
);

select * from customer limit 10;

CREATE TABLE Geolocation (
  zip_cd_prefix varchar(5),
  geoloc_lat varchar(25),
  geoloc_long varchar(25),
  geoloc_city varchar(50),
  geoloc_state varchar(2)
);

select * from geolocation limit 10;

CREATE TABLE Orders (
  order_id varchar(50),
  cust_id varchar(50),
  order_status varchar(20),
  order_purchase_dtm timestamp,
  order_approved_dtm timestamp,
  order_carrier_delivery_dtm timestamp,
  order_cust_delivery_dtm timestamp,
  order_estm_delivery_dtm timestamp
);

select * from orders limit 10;

CREATE TABLE Order_Item (
  order_id varchar(50),
  order_item_id numeric(5),
  product_id varchar(50),
  seller_id varchar(50),
  shipping_limit_dtm timestamp,
  product_price numeric(10,2),
  product_ship_fee numeric(10,2)
);

select * from order_item limit 10;

CREATE TABLE Product_Category (
  category_name varchar(60),
  category_name_en varchar(60)
);

select * from product_category limit 10;

CREATE TABLE Product (
  product_id varchar(50),
  category_name varchar(60),
  name_length numeric(5),
  description_length numeric(10),
  photos_qty numeric(5),
  weight_g numeric(10),
  length_cm numeric(5),
  height_cm numeric(5),
  width_cm numeric(5)
);

select * from product limit 10;

CREATE TABLE Seller (
  seller_id varchar(50),
  seller_zip_cd_prefix varchar(5),
  seller_city varchar(50),
  seller_state varchar(2)
);

select * from seller limit 10;

CREATE TABLE Review (
  review_id varchar(50),
  order_id varchar(50),
  review_score numeric(2),
  review_title text,
  review_message text,
  review_dtm timestamp,
  review_response_dtm timestamp
);

select * from review limit 10;

CREATE TABLE Geolocation_State (
	geoloc_state varchar(2),
	geoloc_state_name varchar(60),
	geoloc_region varchar(20),
	geoloc_state_pop numeric(15,2)
);

select * from geolocation_state;