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

