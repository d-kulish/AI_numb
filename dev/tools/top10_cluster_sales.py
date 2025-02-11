from datetime import datetime, timedelta
from sqlalchemy import text
from langchain_core.tools import tool
from dev.db_config import Session  # Updated import


@tool
def top10_cluster_sales_day(target_date=None):
    """
    Returns the top 10 clusters by sales for a specific day.

    Args:
        target_date (str or None): The target date in 'YYYY-MM-DD' format.
                                 If None, returns yesterday's data.
                                 Only accepts dates within the last 30 days.

    Returns:
        List[Dict]: A list of dictionaries with cluster_id, name, and sales.
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
        WITH daily_cluster_sales AS (
            SELECT 
                cl.cluster_id,
                c.name,
                SUM(s.price) as sales
            FROM am_app_articlesales s
            INNER JOIN am_app_shop sh ON s.shop_id = sh.id
            INNER JOIN am_app_clustershoplink cl ON sh.id = cl.shop_id
            INNER JOIN am_app_cluster c ON cl.cluster_id = c.id
            WHERE DATE(s.date_of_sales) = DATE(:target_date)
            GROUP BY 
                cl.cluster_id,
                c.name
        )
        SELECT *
        FROM daily_cluster_sales
        ORDER BY sales DESC
        LIMIT 10;
        """

        result = session.execute(text(query), {"target_date": target_date})
        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result]

        if not records:
            return {"message": f"No cluster sales data found for date {target_date}"}

        return records

    except Exception as e:
        print(f"Error in top10_cluster_sales_day: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if "session" in locals():
            session.close()
