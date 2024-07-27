# -*- coding: utf-8 -*-
"""
Purpose of this script is to establish a connection to your database 
storing the data.

Reference: Please refer to SQLAlchemy Documentation for understanding
the parameters for the DB connection.

This project work utilized PostgreSQL as a database on my local computer,
so details regarding the username, password, host, and database should
be specific to the user's setup.

"""
import sqlalchemy as db
import pandas as pd
"""
class Connection:
    def __init__(self, customer = pd.DataFrame(), geolocation = pd.DataFrame(),
                 geolocation_state = pd.DataFrame(), order_item = pd.DataFrame(),
                 orders = pd.DataFrame(), product = pd.DataFrame(),
                 product_category = pd.DataFrame(), review = pd.DataFrame(),
                 seller = pd.DataFrame()): 
         self._customer = customer 
         self._geolocation = geolocation
         self._geolocation_state = geolocation_state
         self._order_item = order_item
         self._orders = orders
         self._product = product
         self._product_category = product_category
         self._review = pd.read_sql_table('review', connection)
         self._seller = seller
      
    # getter methods 
    def get_customer(self): 
        return self._customer
    
    def get_geolocation(self):
        return self._geolocation
    
    def get_geolocation_state(self):
        return self._geolocation_state
    
    def get_order_item(self):
        return self._order_item
    
    def get_orders(self):
        return self._orders
    
    def get_product(self):
        return self._product
    
    def get_product_category(self):
        return self._product_category
    
    def get_review(self):
        return self._review
    
    def get_seller(self):
        return self._seller
"""

def db_connect():
    # Return DB credentials
    d = dict()
    d['dbapi'] = input("Enter dialect+driver: ")
    d['username'] = input("Enter username: ")
    d['password'] = input("Enter password: ")
    d['host'] = input("Enter host: ")
    d['database'] = input("Enter database: ")

    url_object = db.URL.create(
        "postgresql",
        username = d['username'],
        password = d['password'],
        host = d['host'],
        database = d['database']
    )
        
    connection = db.create_engine(url_object).connect()
    
    return connection
