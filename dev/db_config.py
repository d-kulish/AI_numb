import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Get all connection parameters from environment variables
db_ip = os.getenv("DB_IP")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
db_port = os.getenv("DB_PORT")

# Create the connection string
connection_string = f"postgresql://{db_user}:{db_password}@{db_ip}:{db_port}/{db_name}"

# Create the SQLAlchemy engine
engine = create_engine(connection_string)
Session = sessionmaker(bind=engine)
