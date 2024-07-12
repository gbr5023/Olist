# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:32:56 2024

@author: gkredila
"""

import sqlalchemy as db
import pandas as pd

# Return DB credentials
def cred():
    d = dict()
    d['dbapi'] = input("Enter dialect+driver: ")
    d['username'] = input("Enter username: ")
    d['password'] = input("Enter password: ")
    d['host'] = input("Enter host: ")
    d['database'] = input("Enter database: ")
    
    return d

d = cred()

url_object = db.URL.create(
    "postgresql",
    username = d['username'],
    password = d['password'],
    host = d['host'],
    database = d['database']
)

connection = db.create_engine(url_object).connect()
#engine = db.create_engine("postgresql://postgres:021818@localhost/postgres")
#conn = engine.connect()
#metadata = db.MetaData()

# Load Tables
customer = pd.read_sql_table('customer', connection)
geolocation = pd.read_sql_table('geolocation', connection)
geolocation_state = pd.read_sql_table('geolocation_state', connection)
order_item = pd.read_sql_table('order_item', connection)
orders = pd.read_sql_table('orders', connection)
product = pd.read_sql_table('product', connection)
product_category = pd.read_sql_table('product_category', connection)
review = pd.read_sql_table('review', connection)
seller = pd.read_sql_table('seller', connection)