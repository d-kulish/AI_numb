from datetime import datetime, timedelta
from sqlalchemy import text
from langchain_core.tools import tool
from dev.db_config import Session


@tool
def top10_cluster_sales_period(end_date=None, period_days=7):
    """
    Returns the top 10 clusters by sales for a specific period.

    Args:
        end_date (str or None): The end date in 'YYYY-MM-DD' format.
                               If None, uses yesterday as end date.
        period_days (int): Number of days to look back (default: 7).
                          Maximum allowed: 30 days.

    Returns:
        List[Dict]: A list of dictionaries with cluster_id, name, and sales.
        Dict: Error message if date is invalid or no data found.
    """
    try:
        today = datetime.now().date()

        # Validate and set period_days
        if not isinstance(period_days, int) or period_days <= 0 or period_days > 30:
            return {"error": "Period must be between 1 and 30 days"}

        # Date handling
        if end_date is None:
            end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        elif isinstance(end_date, str):
            try:
                parsed_end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                if parsed_end_date > today:
                    return {"error": "Cannot query future dates"}
                if parsed_end_date < (today - timedelta(days=30)):
                    return {
                        "error": f"Can only query dates between {(today - timedelta(days=30)).strftime('%Y-%m-%d')} and {today.strftime('%Y-%m-%d')}"
                    }
            except ValueError:
                return {"error": "Invalid date format. Please use YYYY-MM-DD"}

        # Calculate start date
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=period_days - 1)
        ).strftime("%Y-%m-%d")

        session = Session()

        query = """
        WITH period_cluster_sales AS (
            SELECT 
                cl.cluster_id,
                c.name,
                SUM(s.price) as sales
            FROM am_app_articlesales s
            INNER JOIN am_app_shop sh ON s.shop_id = sh.id
            INNER JOIN am_app_clustershoplink cl ON sh.id = cl.shop_id
            INNER JOIN am_app_cluster c ON cl.cluster_id = c.id
            WHERE DATE(s.date_of_sales) BETWEEN DATE(:start_date) AND DATE(:end_date)
            GROUP BY 
                cl.cluster_id,
                c.name
        )
        SELECT *
        FROM period_cluster_sales
        ORDER BY sales DESC
        LIMIT 10;
        """

        result = session.execute(
            text(query), {"start_date": start_date, "end_date": end_date}
        )
        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result]

        if not records:
            return {
                "message": f"No cluster sales data found for period {start_date} to {end_date}"
            }

        return records

    except Exception as e:
        print(f"Error in top10_cluster_sales_period: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if "session" in locals():
            session.close()
