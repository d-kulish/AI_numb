import os
import requests
import pandas as pd
import warnings
from dotenv import load_dotenv
from urllib.parse import quote
from langchain_core.tools import tool

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
load_dotenv()


class SisenseClient:
    """Handles authentication and querying for Sisense."""

    def __init__(self):
        self.sisense_url = os.getenv("SISENSE_URL")
        self.username = os.getenv("SISENSE_USER")
        self.password = os.getenv("SISENSE_PASSWORD")
        self.session = None
        self.authenticated = False

        if not all([self.sisense_url, self.username, self.password]):
            print(
                "Error: Sisense credentials (URL, USER, PASSWORD) not found in environment variables."
            )
            return

        self._authenticate()

    def _authenticate(self):
        """Authenticates with the Sisense API."""
        try:
            self.session = requests.Session()
            self.session.verify = False
            self.session.get(f"{self.sisense_url}/")

            auth_url = f"{self.sisense_url}/api/v1/authentication/login"
            auth_payload = {"username": self.username, "password": self.password}
            auth_response = self.session.post(auth_url, json=auth_payload)

            if auth_response.status_code == 200:
                resp_data = auth_response.json()
                if "access_token" in resp_data:
                    self.session.headers.update(
                        {
                            "Authorization": f"Bearer {resp_data['access_token']}",
                            "Content-Type": "application/json",
                            "accept": "application/json",
                        }
                    )
                    if "csrf_token" in resp_data:
                        # Note: Check if X-XSRF-TOKEN is the correct header name for your Sisense version
                        self.session.headers.update(
                            {"X-XSRF-TOKEN": resp_data["csrf_token"]}
                        )
                    self.authenticated = True
                    print("Sisense authentication successful!")
                else:
                    print(
                        "Authentication succeeded but no access_token found in response."
                    )
            else:
                print(
                    f"Sisense authentication failed: {auth_response.status_code} - {auth_response.text}"
                )
        except requests.exceptions.RequestException as e:
            print(f"Error during Sisense authentication: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during authentication setup: {e}")

    def run_sql_query(
        self, query: str, datasource: str = "RAD_NOVUS_FULL", count: int = -1
    ) -> pd.DataFrame | str:
        """
        Run a SQL query against the specified Sisense datasource.

        Parameters:
        - query: SQL query string.
        - datasource: Name of the Sisense datasource.
        - count: Number of rows to return (-1 for all).

        Returns:
        - Pandas DataFrame with the results or an error message string.
        """
        if not self.authenticated or not self.session:
            return "Error: Sisense client is not authenticated."

        try:
            encoded_query = quote(query)
            endpoint = f"{self.sisense_url}/api/datasources/{datasource}/sql?count={count}&includeMetadata=true&query={encoded_query}"
            response = self.session.get(endpoint)

            if response.status_code == 200:
                data = response.json()

                if "values" in data and "headers" in data:
                    df = pd.DataFrame(data=data["values"], columns=data["headers"])
                    print(
                        f"Sisense query executed successfully. Retrieved {len(df)} rows."
                    )
                    return df
                else:
                    print(f"Sisense query returned unexpected format: {data}")
                    return f"Error: Unexpected response format from Sisense API. Data: {str(data)[:200]}..."  # Return truncated data
            else:
                print(f"Sisense query failed: {response.status_code}")
                print(response.text)
                return f"Error: Sisense query failed with status {response.status_code}. Response: {response.text[:200]}..."  # Return truncated error

        except requests.exceptions.RequestException as e:
            print(f"Error during Sisense query execution: {e}")
            return f"Error: Network or request error during Sisense query: {e}"
        except Exception as e:
            print(f"An unexpected error occurred during query execution: {e}")
            return f"Error: An unexpected error occurred: {e}"


@tool
def all_projects_performance(
    query: str, datasource: str = "C4R_CM_Project_Data_Check_DEV"
) -> str:
    """
    This function returns list of KPIs for all projects - Project ID, Base Sales, Base Units, Total Sales, Total Units, Promo Sales, Promo Share, Back Margin max week and min week.
    It should be used as overal information for all projects or comparative analysis among projects.

    Args:
        table_name (datasource): The name of Sisense cube usef for the dashboard.

    Returns:
        python string with details of each project included into the query result.

    """
    client = SisenseClient()

    table = "sales_check"
    query = f"""
        SELECT project_id as "Project ID", 
                BaseSales as "Base Sales", 
                BaseUnits as "Base Units", 
                TotalSales as "Total Sales", 
                TotalUnits as "Total Units", 
                PromoSales as "Promo Sales",
                PromoShare as "Promo Share",
                maxweek as "Back Margin max week",
                minweek as "Back Margin min week"
        FROM {table} 
        where project_id in (186, 203, 204, 205, 207)"""
    if not client.authenticated:
        return "Failed to authenticate with Sisense. Check credentials and Sisense service status."

    result = client.run_sql_query(query=query, datasource=datasource)

    if isinstance(result, pd.DataFrame):
        # Convert DataFrame to string format suitable for LLM response
        # Adjust max_rows, max_cols, width as needed
        # return result.to_string(max_rows=10, max_cols=10)
        return result.to_string()
    else:
        return result
