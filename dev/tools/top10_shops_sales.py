from datetime import datetime, timedelta
from sqlalchemy import text
from langchain_core.tools import tool
from dev.db_config import Session  # Updated import


@tool
def top10_shops_sales_day(target_date=None):
    """
    Returns the top 10 shops by sales for a specific day.

    Args:
        target_date (str or None): The target date in 'YYYY-MM-DD' format.
                                 If None, returns yesterday's data.
                                 Only accepts dates within the last 30 days.

    Returns:
        List[Dict]: A list of dictionaries with shop_id, physical_address, and sales.
        Dict: Error message if date is invalid or no data found.
    """
    try:
        today = datetime.now().date()

        # Date handling
        if target_date is None:
            target_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        elif isinstance(target_date, str):
            try:
                parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
                # Validate date range
                if parsed_date > today:
                    return {"error": "Cannot query future dates"}
                if parsed_date < (today - timedelta(days=30)):
                    return {
                        "error": f"Can only query dates between {(today - timedelta(days=30)).strftime('%Y-%m-%d')} and {today.strftime('%Y-%m-%d')}"
                    }
            except ValueError:
                return {"error": "Invalid date format. Please use YYYY-MM-DD"}

        session = Session()

        query = """
        WITH daily_shop_sales AS (
            SELECT 
                s.shop_id,
                sh.physical_address,
                COALESCE(SUM(s.price), 0) as sales
            FROM am_app_articlesales s
            INNER JOIN am_app_shop sh 
                ON s.shop_id = sh.id
            WHERE DATE(s.date_of_sales) = DATE(:target_date)
            GROUP BY 
                s.shop_id,
                sh.physical_address
        )
        SELECT *
        FROM daily_shop_sales
        ORDER BY sales DESC
        LIMIT 10;
        """

        result = session.execute(text(query), {"target_date": target_date})
        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result]

        if not records:
            return {"message": f"No shop sales data found for date {target_date}"}

        return records

    except Exception as e:
        print(f"Error in top10_shops_sales_day: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if "session" in locals():
            session.close()
