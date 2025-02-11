from datetime import datetime, timedelta
from sqlalchemy import text
from langchain_core.tools import tool
from dev.db_config import Session  # Updated import


@tool
def top10_product_sales_day(target_date=None):
    """
    Returns the top 10 products by sales for a specific day.

    Args:
        target_date (str or None): The target date in 'YYYY-MM-DD' format.
                                 If None, returns yesterday's data.
                                 Only accepts dates within the last 30 days.

    Returns:
        List[Dict]: A list of dictionaries with product_id, name, and sales.
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
        SELECT 
            aaa.article_id as product_id, 
            aaa2.translation as name, 
            COALESCE(SUM(aaa.price), 0) as sales
        FROM am_app_articlesales aaa
        JOIN am_app_articlestockunitdictionary aaa2 
            ON aaa.article_id = aaa2.id 
        WHERE DATE(aaa.date_of_sales) = DATE(:target_date)
        GROUP BY aaa.article_id, aaa2.translation 
        ORDER BY sales DESC
        LIMIT 10
        """

        result = session.execute(text(query), {"target_date": target_date})
        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result]

        if not records:
            return {"message": f"No sales data found for date {target_date}"}

        return records

    except Exception as e:
        print(f"Error in top10_product_sales_day: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if "session" in locals():
            session.close()
