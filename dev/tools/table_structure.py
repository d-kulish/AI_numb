from sqlalchemy import text
from langchain_core.tools import tool
from dev.db_config import Session  # Updated import


@tool
def table_structure(table_name: str):
    """
    This function returns first 5 rows of any table from database 'am_db'.

    Args:
        table_name (str): The name of the table to query.

    Returns:
        python dictionary with 5 rows from the table.
    """
    session = Session()

    try:
        sql = text(f"SELECT * FROM {table_name} LIMIT 5")
        result = session.execute(sql)
        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result]
    finally:
        session.close()

    return records
