from datetime import datetime, timedelta
from sqlalchemy import text
from langchain_core.tools import tool
from dev.db_config import Session


@tool
def top10_category_sales_day(target_date=None):
    """
    Returns the top 10 categories by sales for a specific day.

    Args:
        target_date (str or None): The target date in 'YYYY-MM-DD' format.
                                 If None, returns yesterday's data.
                                 Only accepts dates within the last 30 days.

    Returns:
        List[Dict]: A list of dictionaries with cat_id, cat_name, and sales.
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
            aac.category_id as cat_id,
            aac2.translation as cat_name,
            SUM(aaa3.price) as sales
        FROM
            am_app_article aaa
            INNER JOIN am_app_articlestockunitdictionary aaa2 ON aaa.id = aaa2.id
            INNER JOIN am_app_articlesales aaa3 ON aaa.id = aaa3.article_id
            INNER JOIN am_app_categorygroupproductlink aac ON aac.group_product_id = aaa.group_product_id
            INNER JOIN am_app_categorydictionary aac2 ON aac.id = aac2.category_id
        WHERE 
            DATE(aaa3.date_of_sales) = DATE(:target_date)
        GROUP BY
            aac.category_id,
            aac2.translation
        ORDER BY
            sales DESC
        LIMIT 10
        """

        result = session.execute(text(query), {"target_date": target_date})
        columns = result.keys()
        records = [dict(zip(columns, row)) for row in result]

        if not records:
            return {"message": f"No category sales data found for date {target_date}"}

        return records

    except Exception as e:
        print(f"Error in top10_category_sales_day: {str(e)}")
        return {"error": f"Database query failed: {str(e)}"}
    finally:
        if "session" in locals():
            session.close()
