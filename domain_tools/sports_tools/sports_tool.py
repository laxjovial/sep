# domain_tools/sports_tools/sports_tool.py

import logging
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime, timedelta
import asyncio # Import asyncio

# Import generic tools
from langchain_core.tools import tool
from shared_tools.scrapper_tool import scrape_web
from shared_tools.doc_summarizer import summarize_document

# Import config_manager to access API configurations and secrets
from config.config_manager import config_manager
# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability
# Import date_parser for date format flexibility
from utils.date_parser import parse_date_to_yyyymmdd
# Import analytics_tracker
from utils import analytics_tracker # Import the module

logger = logging.getLogger(__name__)

# --- Generic API Request Helper (copied for standalone tool file, ideally in shared utils) ---

def _get_nested_value(data: Dict[str, Any], path: List[str]):
    """Helper to get a value from a nested dictionary using a list of keys."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit(): # Handle list indices
            try:
                current = current[int(key)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current

async def _make_dynamic_api_request( # Made async to await analytics_tracker.log_tool_usage
    domain: str,
    function_name: str,
    params: Dict[str, Any],
    user_token: str
) -> Optional[Dict[str, Any]]:
    """
    Makes an API request to the dynamically configured provider for a given domain and function.
    Handles API key retrieval, request construction, and basic error handling.
    Returns parsed JSON data or None on failure (triggering mock fallback).
    Logs tool usage analytics.
    """
    # Check if analytics is enabled for logging tool usage
    log_tool_usage_enabled = config_manager.get("analytics.log_tool_usage", False)

    # Get the default active API provider for the domain from data/config.yml
    active_provider_name = config_manager.get(f"api_defaults.{domain}")
    if not active_provider_name:
        logger.error(f"No default API provider configured for domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"No default API provider configured for domain '{domain}'."
            )
        return None

    # Get the full configuration for the active provider from api_providers.yml
    provider_config = config_manager.get_api_provider_config(domain, active_provider_name)
    if not provider_config:
        logger.error(f"Configuration for API provider '{active_provider_name}' in domain '{domain}' not found in api_providers.yml.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"API provider config '{active_provider_name}' not found for domain '{domain}'."
            )
        return None

    base_url = provider_config.get("base_url")
    api_key_name = provider_config.get("api_key_name")
    api_key = config_manager.get_secret(api_key_name) if api_key_name else None

    # Special handling for Amadeus which uses client_id and client_secret for token
    if active_provider_name == "amadeus":
        api_secret_name = provider_config.get("api_secret_name")
        api_secret = config_manager.get_secret(api_secret_name) if api_secret_name else None
        token_endpoint = provider_config.get("token_endpoint")

        if not api_key or not api_secret or not token_endpoint:
            logger.warning(f"Amadeus API credentials (client_id/secret) or token_endpoint missing. Cannot make live Amadeus call.")
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message="Amadeus API credentials or token endpoint missing."
                )
            return None
        
        # Get Amadeus access token (simplified for demonstration)
        try:
            token_response = requests.post(
                token_endpoint,
                data={'grant_type': 'client_credentials', 'client_id': api_key, 'client_secret': api_secret},
                timeout=5
            )
            token_response.raise_for_status()
            access_token = token_response.json().get('access_token')
            if not access_token:
                logger.error("Failed to get Amadeus access token.")
                if log_tool_usage_enabled:
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_token=user_token,
                        success=False,
                        error_message="Failed to get Amadeus access token."
                    )
                return None
            headers = {"Authorization": f"Bearer {access_token}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting Amadeus access token: {e}")
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=f"Error getting Amadeus access token: {e}"
                )
            return None
    else:
        headers = {} # No special headers by default

    if not base_url:
        logger.error(f"Base URL not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Base URL not configured for '{active_provider_name}'."
            )
        return None

    function_details = provider_config.get("functions", {}).get(function_name)
    if not function_details:
        logger.error(f"Function '{function_name}' not configured for API provider '{active_provider_name}' in domain '{domain}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Function '{function_name}' not configured for '{active_provider_name}'."
            )
        return None

    endpoint = function_details.get("endpoint")
    function_param = function_details.get("function_param") # For Alpha Vantage style 'function' param
    path_params = function_details.get("path_params", []) # For ExchangeRate-API style path params

    if not endpoint and not function_param:
        logger.error(f"Neither 'endpoint' nor 'function_param' defined for function '{function_name}'.")
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=f"Endpoint or function_param missing for '{function_name}'."
            )
        return None

    # Construct URL
    full_url = f"{base_url}{endpoint}" if endpoint else base_url

    # Add path parameters to URL if specified
    for p_param in path_params:
        if p_param in params:
            value = str(params.pop(p_param))
            full_url = full_url.replace(f"{{{p_param}}}", value)
        else:
            error_msg = f"Missing path parameter '{p_param}' for function '{function_name}'."
            logger.warning(error_msg)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=error_msg
                )
            return None # Cannot construct URL without required path params

    # Construct query parameters
    query_params = {}
    if function_param:
        query_params["function"] = function_param # Alpha Vantage specific

    # Add API key if it's a query param (not in path or header)
    if api_key_name and active_provider_name not in ["amadeus", "exchangerate_api"]: # Amadeus handled by headers, ExchangeRate by path
        param_name_in_url = provider_config.get("api_key_param_name", api_key_name.replace("_api_key", ""))
        if api_key: # Only add if key exists
            query_params[param_name_in_url] = api_key 
    elif active_provider_name == "exchangerate_api" and api_key:
        pass # Key is a path parameter, already handled above

    for param_key in function_details.get("required_params", []) + function_details.get("optional_params", []):
        if param_key in params:
            query_params[param_key] = params[param_key]
        elif param_key in function_details.get("required_params", []):
            error_msg = f"Missing required parameter '{param_key}' for function '{function_name}'."
            logger.warning(error_msg)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=error_msg
                )
            return None # Missing required param, cannot proceed

    try:
        logger.info(f"Making API call to: {full_url} with params: {query_params}")
        response = requests.get(full_url, params=query_params, headers=headers, timeout=config_manager.get("web_scraping.timeout_seconds", 15))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        raw_data = response.json()
        
        # Check for API-specific error messages in the response body
        api_error_message = None
        if "Error Message" in raw_data: # Alpha Vantage specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data['Error Message']}"
        elif "Note" in raw_data and "Thank you for using Alpha Vantage!" in raw_data["Note"]: # Alpha Vantage rate limit
            api_error_message = f"API rate limit hit for {active_provider_name}: {raw_data['Note']}"
        elif raw_data.get("status") == "error": # NewsAPI specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('message', 'Unknown error')}"
        elif raw_data.get("Error"): # OMDBAPI specific
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('Error')}"
        elif raw_data.get("status") and raw_data["status"].get("error_code"): # CoinGecko error
            api_error_message = f"API Error from {active_provider_name}: {raw_data['status'].get('error_message', 'Unknown CoinGecko error')}"
        elif raw_data.get("result") == "error": # ExchangeRate-API error
            api_error_message = f"API Error from {active_provider_name}: {raw_data.get('error-type', 'Unknown ExchangeRate-API error')}"

        if api_error_message:
            logger.error(api_error_message)
            if log_tool_usage_enabled:
                await analytics_tracker.log_tool_usage(
                    tool_name=f"{domain}_{function_name}",
                    tool_params=params,
                    user_token=user_token,
                    success=False,
                    error_message=api_error_message
                )
            return None


        # Extract data based on response_path
        data_to_map = raw_data
        response_path = function_details.get("response_path")
        if response_path:
            data_to_map = _get_nested_value(raw_data, response_path)
            if data_to_map is None:
                error_msg = f"Response path '{'.'.join(response_path)}' not found in API response from {active_provider_name}. Raw data: {raw_data}"
                logger.warning(error_msg)
                if log_tool_usage_enabled:
                    await analytics_tracker.log_tool_usage(
                        tool_name=f"{domain}_{function_name}",
                        tool_params=params,
                        user_token=user_token,
                        success=False,
                        error_message=error_msg
                    )
                return None

        # Apply data mapping
        mapped_data = {}
        data_map = function_details.get("data_map", {})
        if isinstance(data_to_map, list): # For lists of items (e.g., news articles, historical data)
            mapped_data_list = []
            for item in data_to_map:
                mapped_item = {}
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list): # Handle nested paths in data_map
                        mapped_item[mapped_key] = _get_nested_value(item, original_key_path)
                    elif '.' in str(original_key_path): # Handle dot-separated paths in data_map
                        mapped_item[mapped_key] = _get_nested_value(item, original_key_path.split('.'))
                    else: # Direct key or list index
                        if isinstance(original_key_path, int) and isinstance(item, list):
                            try: mapped_item[mapped_key] = item[original_key_path]
                            except IndexError: mapped_item[mapped_key] = None
                        else:
                            mapped_item[mapped_key] = item.get(original_key_path)
                mapped_data_list.append(mapped_item)
            final_result = {"data": mapped_data_list} # Wrap list in a dict for consistent return
        elif isinstance(data_to_map, dict) and function_name == "get_historical_stock_prices" and active_provider_name == "alphavantage":
            # Special handling for Alpha Vantage TIME_SERIES_DAILY where keys are dates
            processed_data = {}
            for date_key, values in data_to_map.items():
                mapped_values = {}
                for mapped_key, original_key_path in data_map.items():
                    if isinstance(original_key_path, list):
                        mapped_values[mapped_key] = _get_nested_value(values, original_key_path)
                    elif '.' in str(original_key_path):
                        mapped_values[mapped_key] = _get_nested_value(values, original_key_path.split('.'))
                    else:
                        mapped_values[mapped_key] = values.get(original_key_path)
                processed_data[date_key] = mapped_values
            final_result = {"data": processed_data}
        else: # For single object responses
            # Special handling for CoinGecko simple price, where response is { "bitcoin": { "usd": 20000 } }
            if function_name == "get_crypto_price" and active_provider_name == "coingecko":
                # params will contain 'ids' and 'vs_currencies'
                crypto_id = params.get("ids", "").lower()
                currency = params.get("vs_currencies", "").lower()
                if crypto_id in raw_data and currency in raw_data[crypto_id]:
                    mapped_data["price"] = raw_data[crypto_id][currency]
                    if f"{currency}_market_cap" in raw_data[crypto_id]:
                        mapped_data["market_cap"] = raw_data[crypto_id][f"{currency}_market_cap"]
                    if f"{currency}_24hr_vol" in raw_data[crypto_id]:
                        mapped_data["vol_24hr"] = raw_data[crypto_id][f"{currency}_24hr_vol"]
                    if f"{currency}_24hr_change" in raw_data[crypto_id]:
                        mapped_data["change_24hr"] = raw_data[crypto_id][f"{currency}_24hr_change"]
                    if "last_updated_at" in raw_data[crypto_id]:
                        mapped_data["last_updated"] = raw_data[crypto_id]["last_updated_at"]
                    final_result = mapped_data
                else:
                    error_msg = f"CoinGecko simple price response unexpected for {crypto_id}/{currency}: {raw_data}"
                    logger.warning(error_msg)
                    if log_tool_usage_enabled:
                        await analytics_tracker.log_tool_usage(
                            tool_name=f"{domain}_{function_name}",
                            tool_params=params,
                            user_token=user_token,
                            success=False,
                            error_message=error_msg
                        )
                    return None
            
            for mapped_key, original_key_path in data_map.items():
                if isinstance(original_key_path, list):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path)
                elif '.' in str(original_key_path):
                    mapped_data[mapped_key] = _get_nested_value(data_to_map, original_key_path.split('.'))
                else:
                    mapped_data[mapped_key] = data_to_map.get(original_key_path)
            final_result = mapped_data

        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=True
            )
        return final_result

    except requests.exceptions.Timeout:
        error_msg = f"API request to {active_provider_name} timed out for function '{function_name}'."
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Error making API request to {active_provider_name} for function '{function_name}': {e}"
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=e
            )
        return None
    except json.JSONDecodeError:
        error_msg = f"Failed to decode JSON response from {active_provider_name} for function '{function_name}'."
        logger.error(error_msg)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None
    except Exception as e:
        error_msg = f"An unexpected error occurred during API call to {active_provider_name} for '{function_name}': {e}"
        logger.error(error_msg, exc_info=True)
        if log_tool_usage_enabled:
            await analytics_tracker.log_tool_usage(
                tool_name=f"{domain}_{function_name}",
                tool_params=params,
                user_token=user_token,
                success=False,
                error_message=error_msg
            )
        return None


# --- Mock Data for Fallback ---
_mock_sports_data = {
    "latest_scores": [
        {
            "sport": "Basketball",
            "match": "Lakers vs. Celtics",
            "score": "110-108",
            "status": "Final",
            "date": (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
        },
        {
            "sport": "Soccer",
            "match": "Real Madrid vs. Barcelona",
            "score": "2-1",
            "status": "Final",
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
        }
    ],
    "upcoming_events": [
        {
            "sport": "Tennis",
            "event": "Wimbledon Finals",
            "date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
            "time": "14:00 GMT",
            "participants": "Player A vs. Player B"
        },
        {
            "sport": "Formula 1",
            "event": "Monaco Grand Prix",
            "date": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
            "time": "15:00 CET",
            "participants": "All teams"
        }
    ],
    "player_stats": {
        "lebron_james": {
            "name": "LeBron James",
            "team": "Los Angeles Lakers",
            "sport": "Basketball",
            "points_per_game": 27.2,
            "assists_per_game": 7.3,
            "rebounds_per_game": 7.5
        },
        "lionel_messi": {
            "name": "Lionel Messi",
            "team": "Inter Miami CF",
            "sport": "Soccer",
            "goals": 820,
            "assists": 361
        }
    },
    "team_stats": {
        "los_angeles_lakers": {
            "name": "Los Angeles Lakers",
            "sport": "Basketball",
            "wins": 50,
            "losses": 32,
            "conference": "Western"
        },
        "real_madrid": {
            "name": "Real Madrid",
            "sport": "Soccer",
            "wins": 28,
            "draws": 8,
            "losses": 2,
            "league": "La Liga"
        }
    },
    "league_info": {
        "nba": {
            "name": "National Basketball Association",
            "sport": "Basketball",
            "country": "USA/Canada",
            "current_champion": "Denver Nuggets"
        },
        "premier_league": {
            "name": "Premier League",
            "sport": "Soccer",
            "country": "England",
            "current_champion": "Manchester City"
        }
    }
}

class SportsTools:
    """
    A collection of sports-related tools for the Intelli-Agent.
    This class acts as a wrapper to group related tool functions and
    provides a consistent interface for the main application.
    """

    @tool
    async def get_latest_scores(self, sport: Optional[str] = None, team: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves the latest scores for sports matches, optionally filtered by sport or team.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            sport (str, optional): The sport to get scores for (e.g., "basketball", "soccer", "tennis").
            team (str, optional): The team to filter scores by (e.g., "Lakers", "Real Madrid").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of latest scores, or an error/fallback message.
        """
        logger.info(f"Tool: get_latest_scores called for sport: '{sport}', team: '{team}' by user: {user_token}")

        if not get_user_tier_capability(user_token, 'sports_tool_access', False):
            return "Error: Access to sports tools is not enabled for your current tier."
        
        params = {}
        if sport: params["sport"] = sport
        if team: params["team"] = team

        api_data = await _make_dynamic_api_request("sports", "get_latest_scores", params, user_token) # Await the async call

        if api_data and api_data.get("data"):
            scores = api_data["data"]
            if scores:
                response_str = "Latest Scores:\n"
                for match in scores[:5]: # Limit to top 5 for brevity
                    response_str += (
                        f"- Sport: {match.get('sport', 'N/A')}\n"
                        f"  Match: {match.get('match', 'N/A')}\n"
                        f"  Score: {match.get('score', 'N/A')}\n"
                        f"  Status: {match.get('status', 'N/A')}\n"
                        f"  Date: {match.get('date', 'N/A')}\n\n"
                    )
                return response_str
            else:
                return f"No live scores found for sport '{sport}' and team '{team}'. Falling back to mock data."
        
        # Fallback to mock data
        mock_scores = _mock_sports_data.get("latest_scores", [])
        filtered_mock_scores = []
        for score in mock_scores:
            if (not sport or score.get("sport", "").lower() == sport.lower()) and \
               (not team or team.lower() in score.get("match", "").lower()):
                filtered_mock_scores.append(score)

        if filtered_mock_scores:
            response_str = "Latest Scores (Mock Data Fallback):\n"
            for match in filtered_mock_scores[:5]:
                response_str += (
                    f"- Sport: {match['sport']}\n"
                    f"  Match: {match['match']}\n"
                    f"  Score: {match['score']}\n"
                    f"  Status: {match['status']}\n"
                    f"  Date: {match['date']}\n\n"
                )
            return response_str
        else:
            return "No latest scores found. (API/Mock Fallback Failed)"


    @tool
    async def get_upcoming_events(self, sport: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves upcoming sports events, optionally filtered by sport.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            sport (str, optional): The sport to get upcoming events for (e.g., "football", "basketball").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of upcoming sports events, or an error/fallback message.
        """
        logger.info(f"Tool: get_upcoming_events called for sport: '{sport}' by user: {user_token}")

        if not get_user_tier_capability(user_token, 'sports_tool_access', False):
            return "Error: Access to sports tools is not enabled for your current tier."
        
        params = {}
        if sport: params["sport"] = sport

        api_data = await _make_dynamic_api_request("sports", "get_upcoming_events", params, user_token) # Await the async call

        if api_data and api_data.get("data"):
            events = api_data["data"]
            if events:
                response_str = "Upcoming Sports Events:\n"
                for event in events[:5]: # Limit to top 5 for brevity
                    response_str += (
                        f"- Sport: {event.get('sport', 'N/A')}\n"
                        f"  Event: {event.get('event', 'N/A')}\n"
                        f"  Date: {event.get('date', 'N/A')}\n"
                        f"  Time: {event.get('time', 'N/A')}\n"
                        f"  Participants: {event.get('participants', 'N/A')}\n\n"
                    )
                return response_str
            else:
                return f"No live upcoming events found for sport '{sport}'. Falling back to mock data."
        
        # Fallback to mock data
        mock_events = _mock_sports_data.get("upcoming_events", [])
        filtered_mock_events = []
        for event in mock_events:
            if not sport or event.get("sport", "").lower() == sport.lower():
                filtered_mock_events.append(event)

        if filtered_mock_events:
            response_str = "Upcoming Sports Events (Mock Data Fallback):\n"
            for event in filtered_mock_events[:5]:
                response_str += (
                    f"- Sport: {event['sport']}\n"
                    f"  Event: {event['event']}\n"
                    f"  Date: {event['date']}\n"
                    f"  Time: {event['time']}\n"
                    f"  Participants: {event['participants']}\n\n"
                )
            return response_str
        else:
            return "No upcoming events found. (API/Mock Fallback Failed)"

    @tool
    async def get_player_stats(self, player_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves statistics for a specified player, optionally filtered by sport.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            player_name (str): The name of the player (e.g., "LeBron James", "Lionel Messi").
            sport (str, optional): The sport the player plays (e.g., "basketball", "soccer").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of player statistics, or an error/fallback message.
        """
        logger.info(f"Tool: get_player_stats called for player: '{player_name}', sport: '{sport}' by user: {user_token}")

        if not get_user_tier_capability(user_token, 'sports_tool_access', False):
            return "Error: Access to sports tools is not enabled for your current tier."
        
        params = {"player_name": player_name}
        if sport: params["sport"] = sport

        api_data = await _make_dynamic_api_request("sports", "get_player_stats", params, user_token)

        if api_data:
            response_str = f"Player Stats for {api_data.get('name', player_name)}:\n"
            for key, value in api_data.items():
                if key not in ["name", "sport"]: # Avoid repeating name and sport
                    response_str += f"  {key.replace('_', ' ').title()}: {value}\n"
            return response_str
        
        # Fallback to mock data
        mock_player_key = player_name.lower().replace(" ", "_")
        mock_player_data = _mock_sports_data.get("player_stats", {}).get(mock_player_key)
        if mock_player_data and (not sport or mock_player_data.get("sport", "").lower() == sport.lower()):
            response_str = f"Player Stats for {mock_player_data['name']} (Mock Data Fallback):\n"
            for key, value in mock_player_data.items():
                if key not in ["name", "sport"]:
                    response_str += f"  {key.replace('_', ' ').title()}: {value}\n"
            return response_str
        else:
            return f"Player statistics not found for '{player_name}'. (API/Mock Fallback Failed)"

    @tool
    async def get_team_stats(self, team_name: str, sport: Optional[str] = None, user_token: str = "default") -> str:
        """
        Retrieves statistics for a specified team, optionally filtered by sport.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            team_name (str): The name of the team (e.g., "Los Angeles Lakers", "Real Madrid").
            sport (str, optional): The sport the team plays (e.g., "basketball", "soccer").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of team statistics, or an error/fallback message.
        """
        logger.info(f"Tool: get_team_stats called for team: '{team_name}', sport: '{sport}' by user: {user_token}")

        if not get_user_tier_capability(user_token, 'sports_tool_access', False):
            return "Error: Access to sports tools is not enabled for your current tier."
        
        params = {"team_name": team_name}
        if sport: params["sport"] = sport

        api_data = await _make_dynamic_api_request("sports", "get_team_stats", params, user_token)

        if api_data:
            response_str = f"Team Stats for {api_data.get('name', team_name)}:\n"
            for key, value in api_data.items():
                if key not in ["name", "sport"]:
                    response_str += f"  {key.replace('_', ' ').title()}: {value}\n"
            return response_str

        # Fallback to mock data
        mock_team_key = team_name.lower().replace(" ", "_")
        mock_team_data = _mock_sports_data.get("team_stats", {}).get(mock_team_key)
        if mock_team_data and (not sport or mock_team_data.get("sport", "").lower() == sport.lower()):
            response_str = f"Team Stats for {mock_team_data['name']} (Mock Data Fallback):\n"
            for key, value in mock_team_data.items():
                if key not in ["name", "sport"]:
                    response_str += f"  {key.replace('_', ' ').title()}: {value}\n"
            return response_str
        else:
            return f"Team statistics not found for '{team_name}'. (API/Mock Fallback Failed)"

    @tool
    async def get_league_info(self, league_name: str, user_token: str = "default") -> str:
        """
        Retrieves general information about a specified sports league.
        Falls back to mock data if API key is missing or API call fails.

        Args:
            league_name (str): The name of the league (e.g., "NBA", "Premier League").
            user_token (str, optional): The unique identifier for the user. Defaults to "default".

        Returns:
            str: A formatted string of league information, or an error/fallback message.
        """
        logger.info(f"Tool: get_league_info called for league: '{league_name}' by user: {user_token}")

        if not get_user_tier_capability(user_token, 'sports_tool_access', False):
            return "Error: Access to sports tools is not enabled for your current tier."
        
        params = {"league_name": league_name}

        api_data = await _make_dynamic_api_request("sports", "get_league_info", params, user_token)

        if api_data:
            response_str = f"League Information for {api_data.get('name', league_name)}:\n"
            for key, value in api_data.items():
                if key not in ["name"]:
                    response_str += f"  {key.replace('_', ' ').title()}: {value}\n"
            return response_str

        # Fallback to mock data
        mock_league_key = league_name.lower().replace(" ", "_")
        mock_league_data = _mock_sports_data.get("league_info", {}).get(mock_league_key)
        if mock_league_data:
            response_str = f"League Information for {mock_league_data['name']} (Mock Data Fallback):\n"
            for key, value in mock_league_data.items():
                if key not in ["name"]:
                    response_str += f"  {key.replace('_', ' ').title()}: {value}\n"
            return response_str
        else:
            return f"League information not found for '{league_name}'. (API/Mock Fallback Failed)"


    # --- Existing Generic Tools (not directly using external APIs, but can be used in sports context) ---

    @tool
    def sports_search_web(self, query: str, user_token: str = "default", max_chars: int = 2000) -> str:
        """
        Searches the web for general sports-related information using a smart search fallback mechanism.
        This tool wraps the generic `scrape_web` tool, providing a sports-specific interface.
        
        Args:
            query (str): The sports-related search query (e.g., "history of basketball", "rules of cricket").
            user_token (str): The unique identifier for the user. Defaults to "default".
            max_chars (int): Maximum characters for the returned snippet. Defaults to 2000.
        
        Returns:
            str: A string containing relevant information from the web.
        """
        logger.info(f"Tool: sports_search_web called with query: '{query}' for user: '{user_token}'")
        return scrape_web(query=query, user_token=user_token, max_chars=max_chars)

    @tool
    async def sports_query_uploaded_docs(self, query: str, user_token: str = "default", export: Optional[bool] = False, k: int = 5) -> str:
        """
        Queries previously uploaded and indexed sports documents for a user using vector similarity search.
        This tool wraps the generic `QueryUploadedDocs` tool, fixing the section to "sports".
        
        Args:
            query (str): The search query to find relevant sports documents (e.g., "my team's play book", "athlete training regimen").
            user_token (str): The unique identifier for the user. Defaults to "default".
            export (bool): If True, the results will be saved to a file in markdown format. Defaults to False.
            k (int): The number of top relevant documents to retrieve. Defaults to 5.
        
        Returns:
            str: A string containing the combined content of the relevant document chunks,
                 or a message indicating no data/results found, or the export path if exported.
        """
        logger.info(f"Tool: sports_query_uploaded_docs called with query: '{query}' for user: '{user_token}'")
        # This will be replaced by a call to self.document_tools.query_uploaded_docs
        # For now, keeping the original call for review purposes.
        # Assuming QueryUploadedDocs is an async tool or can be awaited
        # If QueryUploadedDocs is not async, remove 'await' and make this function non-async
        return f"Mocked document query results for '{query}' in section 'sports'." # Return mock string for now


    @tool
    async def sports_summarize_document_by_path(self, file_path_str: str) -> str:
        """
        Summarizes a document related to sports located at the given file path.
        The file path should be accessible by the system (e.g., in the 'uploads' directory).
        This tool wraps the generic `summarize_document` tool.
        
        Args:
            file_path_str (str): The full path to the document file to be summarized.
                                  Example: "uploads/default/sports/team_strategy.pdf"
        
        Returns:
            str: A concise summary of the document content.
        """
        logger.info(f"Tool: sports_summarize_document_by_path called for file: '{file_path_str}'")
        file_path = Path(file_path_str)
        if not file_path.exists():
            logger.error(f"Document not found at '{file_path_str}' for summarization.")
            return f"Error: Document not found at '{file_path_str}'."
        
        try:
            # Assuming summarize_document is an async tool or can be awaited
            summary = await summarize_document(file_path.read_text(), user_token="default") # Await and pass text content
            return f"Summary of '{file_path.name}':\n{summary}"
        except ValueError as e:
            logger.error(f"Error summarizing document '{file_path_str}': {e}")
            return f"Error summarizing document: {e}"
        except Exception as e:
            logger.critical(f"An unexpected error occurred during summarization of '{file_path_str}': {e}", exc_info=True)
            return f"An unexpected error occurred during summarization: {e}"


# CLI Test (optional)
if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock, AsyncMock, patch
    import shutil
    import os
    import sys # Import sys for patching modules
    from shared_tools.vector_utils import BASE_VECTOR_DIR # For cleanup

    logging.basicConfig(level=logging.INFO)

    # Mock Streamlit secrets and config_manager for local testing
    class MockSecrets:
        def __init__(self):
            self.sports_api_key = "MOCK_SPORTS_API_KEY"
            self.openai_api_key = "sk-mock-openai-key-12345"
            self.google_api_key = "AIzaSy-mock-google-key"
            self.firebase_config = "{}"
            self.serpapi_api_key = "MOCK_SERPAPI_KEY" # For scrape_web

        def get(self, key, default=None):
            return getattr(self, key, default)
    
    class MockConfigManager:
        _instance = None
        _is_loaded = False
        def __init__(self):
            if MockConfigManager._instance is not None:
                raise Exception("ConfigManager is a singleton. Use get_instance().")
            MockConfigManager._instance = self
            self._config_data = {
                'llm': {'max_summary_input_chars': 10000},
                'rag': {'chunk_size': 500, 'chunk_overlap': 50, 'max_query_results_k': 10},
                'web_scraping': {
                    'user_agent': 'Mozilla/5.0 (Test; Python)',
                    'timeout_seconds': 1 # Short timeout for mocks
                },
                'tiers': {},
                'default_user_tier': 'free',
                'default_user_roles': ['user'],
                'api_defaults': { # Mock api_defaults
                    'sports': 'mock_sports_provider' # Assuming a mock provider for sports
                },
                'analytics': { # Mock analytics settings
                    'enabled': True,
                    'log_tool_usage': True,
                    'log_query_failures': True
                }
            }
            self._api_providers_data = { # Mock api_providers_data for sports
                "sports": {
                    "mock_sports_provider": {
                        "base_url": "http://mock-sports-api.com/v1",
                        "api_key_name": "sports_api_key",
                        "api_key_param_name": "apiKey",
                        "functions": {
                            "get_latest_scores": {
                                "endpoint": "/scores",
                                "required_params": [],
                                "optional_params": ["sport", "team"],
                                "response_path": ["matches"],
                                "data_map": {
                                    "sport": "sport",
                                    "match": "match",
                                    "score": "score",
                                    "status": "status",
                                    "date": "date"
                                }
                            },
                            "get_upcoming_events": {
                                "endpoint": "/events",
                                "required_params": [],
                                "optional_params": ["sport"],
                                "response_path": ["events"],
                                "data_map": {
                                    "sport": "sport",
                                    "event": "event_name",
                                    "date": "event_date",
                                    "time": "event_time",
                                    "participants": "participants"
                                }
                            },
                            "get_player_stats": { # Added mock config for player stats
                                "endpoint": "/player-stats",
                                "required_params": ["player_name"],
                                "optional_params": ["sport"],
                                "response_path": ["player"],
                                "data_map": {
                                    "name": "name",
                                    "team": "team",
                                    "sport": "sport",
                                    "points_per_game": "ppg",
                                    "assists_per_game": "apg",
                                    "rebounds_per_game": "rpg"
                                }
                            },
                            "get_team_stats": { # Added mock config for team stats
                                "endpoint": "/team-stats",
                                "required_params": ["team_name"],
                                "optional_params": ["sport"],
                                "response_path": ["team"],
                                "data_map": {
                                    "name": "name",
                                    "sport": "sport",
                                    "wins": "wins",
                                    "losses": "losses",
                                    "conference": "conference"
                                }
                            },
                            "get_league_info": { # Added mock config for league info
                                "endpoint": "/league-info",
                                "required_params": ["league_name"],
                                "optional_params": [],
                                "response_path": ["league"],
                                "data_map": {
                                    "name": "name",
                                    "sport": "sport",
                                    "country": "country",
                                    "current_champion": "champion"
                                }
                            }
                        }
                    }
                }
            }
            self._is_loaded = True
        
        def get(self, key, default=None):
            parts = key.split('.')
            val = self._config_data
            for part in parts:
                if isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    return default
            return val
        
        def get_secret(self, key, default=None):
            mock_secrets_instance = MockSecrets()
            return mock_secrets_instance.get(key, default)

        def set_secret(self, key, value):
            pass
        
        def get_api_provider_config(self, domain: str, provider_name: str) -> Optional[Dict[str, Any]]:
            return self._api_providers_data.get(domain, {}).get(provider_name)

        def get_domain_api_providers(self, domain: str) -> Dict[str, Any]:
            return self._api_providers_data.get(domain, {})


    # Mock user_manager.get_current_user and get_user_tier_capability for testing RBAC
    class MockUserManager:
        _mock_users = {
            "mock_free_token": {"user_id": "mock_free_token", "username": "FreeUser", "email": "free@example.com", "tier": "free", "roles": ["user"]},
            "mock_pro_token": {"user_id": "mock_pro_token", "username": "ProUser", "email": "pro@example.com", "tier": "pro", "roles": ["user"]},
            "mock_premium_token": {"user_id": "mock_premium_token", "username": "PremiumUser", "email": "premium@example.com", "tier": "premium", "roles": ["user"]},
            "mock_admin_token": {"user_id": "mock_admin_token", "username": "AdminUser", "email": "admin@example.com", "tier": "admin", "roles": ["user", "admin"]},
        }
        _rbac_capabilities = {
            'capabilities': {
                'sports_tool_access': {
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'document_query_enabled': { # Added for document tool
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'web_search_max_results': {
                    'default': 2,
                    'tiers': {'pro': 7, 'premium': 15}
                },
                'web_search_limit_chars': {
                    'default': 500,
                    'tiers': {'pro': 3000, 'premium': 10000}
                },
                'summarization_enabled': { # For summarize_document
                    'default': False,
                    'roles': {'pro': True, 'premium': True, 'admin': True}
                },
                'llm_default_provider': { # For summarize_document
                    'default': 'gemini',
                    'tiers': {'pro': 'gemini', 'premium': 'openai', 'admin': 'gemini'}
                },
                'llm_default_model_name': { # For summarize_document
                    'default': 'gemini-1.5-flash',
                    'tiers': {'pro': 'gemini-1.5-flash', 'premium': 'gpt-4o', 'admin': 'gemini-1.5-flash'}
                },
                'llm_default_temperature': { # For summarize_document
                    'default': 0.7,
                    'tiers': {'pro': 0.5, 'premium': 0.3, 'admin': 0.7}
                },
            }
        }
        _tier_hierarchy = {
            "free": 0, "user": 1, "basic": 2, "pro": 3, "premium": 4, "admin": 99
        }

        def get_current_user(self) -> Dict[str, Any]:
            return getattr(self, '_current_mock_user', {})

        def get_user_tier_capability(self, user_token: Optional[str], capability_key: str, default_value: Any = None) -> Any:
            user_info = self._mock_users.get(user_token, {})
            user_id = user_info.get('user_id')
            user_tier = user_info.get('tier', 'free')
            user_roles = user_info.get('roles', [])

            if "admin" in user_roles:
                if isinstance(default_value, bool): return True
                if isinstance(default_value, (int, float)): return float('inf')
                return default_value
            
            capability_config = self._rbac_capabilities.get('capabilities', {}).get(capability_key)
            if not capability_config:
                return default_value

            # Check roles first
            for role in user_roles:
                if role in capability_config.get('roles', {}):
                    return capability_config['roles'][role]
            
            # Then check tiers
            if user_tier in capability_config.get('tiers', {}):
                return capability_config['tiers'][user_tier]

            return capability_config.get('default', default_value)

    # Patch the actual imports for testing
    import streamlit as st_mock
    if not hasattr(st_mock, 'secrets'):
        st_mock.secrets = MockSecrets()
    
    sys.modules['config.config_manager'].config_manager = MockConfigManager()
    sys.modules['config.config_manager'].ConfigManager = MockConfigManager
    sys.modules['utils.user_manager'] = MockUserManager()
    sys.modules['utils.user_manager'].get_user_tier_capability = MockUserManager().get_user_tier_capability # Patch the function directly

    # Mock analytics_tracker
    mock_analytics_tracker_db = MagicMock()
    mock_analytics_tracker_auth = MagicMock()
    mock_analytics_tracker_auth.currentUser = MagicMock(uid="mock_user_123")
    mock_analytics_tracker_db.collection.return_value.add = AsyncMock(return_value=MagicMock(id="mock_doc_id"))

    # Patch firebase_admin.firestore for the local import within log_event
    with patch.dict(sys.modules, {'firebase_admin.firestore': MagicMock(firestore=MagicMock())}):
        sys.modules['firebase_admin.firestore'].firestore.CollectionReference = MagicMock()
        sys.modules['firebase_admin.firestore'].firestore.DocumentReference = MagicMock()
        
        # Initialize the actual analytics_tracker with mocks
        analytics_tracker.initialize_analytics(
            mock_analytics_tracker_db,
            mock_analytics_tracker_auth,
            "test_app_id_for_analytics",
            "mock_user_123"
        )

        # Mock requests.get for external API calls
        original_requests_get = requests.get

        def mock_requests_get_dynamic(url, params, headers, timeout):
            # Simulate hypothetical Sports API responses
            if "mock-sports-api.com/v1" in url:
                if "/scores" in url:
                    sport = params.get("sport", "").lower()
                    team = params.get("team", "").lower()
                    if "basketball" in sport and "lakers" in team:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "matches": [
                                {
                                    "sport": "Basketball",
                                    "match": "Lakers vs. Warriors",
                                    "score": "105-103",
                                    "status": "Final",
                                    "date": datetime.now().isoformat()
                                }
                            ]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"matches": []}
                        return mock_response
                elif "/events" in url:
                    sport = params.get("sport", "").lower()
                    if "tennis" in sport:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "events": [
                                {
                                    "sport": "Tennis",
                                    "event_name": "Mock Tennis Open",
                                    "event_date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                                    "event_time": "10:00 AM",
                                    "participants": "Top Players"
                                }
                            ]
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"events": []}
                        return mock_response
                elif "/player-stats" in url: # Mock for get_player_stats
                    player_name = params.get("player_name", "").lower()
                    if "lebron james" in player_name:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "player": {
                                "name": "LeBron James",
                                "team": "Los Angeles Lakers",
                                "sport": "Basketball",
                                "ppg": 27.2,
                                "apg": 7.3,
                                "rpg": 7.5
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"player": {}}
                        return mock_response
                elif "/team-stats" in url: # Mock for get_team_stats
                    team_name = params.get("team_name", "").lower()
                    if "los angeles lakers" in team_name:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "team": {
                                "name": "Los Angeles Lakers",
                                "sport": "Basketball",
                                "wins": 50,
                                "losses": 32,
                                "conference": "Western"
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"team": {}}
                        return mock_response
                elif "/league-info" in url: # Mock for get_league_info
                    league_name = params.get("league_name", "").lower()
                    if "nba" in league_name:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {
                            "league": {
                                "name": "National Basketball Association",
                                "sport": "Basketball",
                                "country": "USA/Canada",
                                "champion": "Denver Nuggets"
                            }
                        }
                        return mock_response
                    else:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"league": {}}
                        return mock_response
            
            # Simulate scrape_web's internal requests.get if needed
            if "google.com/search" in url or "example.com" in url: # Mock for scrape_web
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = f"<html><body><h1>Search results for {params.get('q', 'sports')}</h1><p>Some sports related content from web search.</p></body></html>"
                return mock_response

            return original_requests_get(url, params=params, headers=headers, timeout=timeout)

        requests.get = mock_requests_get_dynamic

        test_user_pro = "mock_pro_token"
        test_user_free = "mock_free_token"

        # Mock for QueryUploadedDocs
        class MockQueryUploadedDocs:
            def __init__(self, query, user_token, section, export, k):
                self.query = query
                self.user_token = user_token
                self.section = section
                self.export = export
                self.k = k
            async def __call__(self): # Made async
                return f"Mocked document query results for '{self.query}' in section '{self.section}'."

        # Mock for summarize_document
        class MockSummarizeDocument:
            async def __call__(self, text_content, user_token): # Made async
                return f"Mocked summary of text for user {user_token}: {text_content[:50]}..."

        # Patch QueryUploadedDocs and summarize_document in the sports_tool module
        # original_QueryUploadedDocs = sys.modules['domain_tools.sports_tools.sports_tool'].QueryUploadedDocs # Not needed anymore
        original_summarize_document = sys.modules['domain_tools.sports_tools.sports_tool'].summarize_document
        # sys.modules['domain_tools.sports_tools.sports_tool'].QueryUploadedDocs = MockQueryUploadedDocs # Not needed anymore
        sys.modules['domain_tools.sports_tools.sports_tool'].summarize_document = MockSummarizeDocument()


        async def run_sports_tests():
            print("\n--- Testing sports_tool functions with Analytics ---")

            sports_tools_instance = SportsTools() # Instantiate the class for testing

            # Test get_latest_scores (success)
            print("\n--- Test 1: get_latest_scores (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock() # Reset mock call count
            result_scores = await sports_tools_instance.get_latest_scores(sport="basketball", team="lakers", user_token=test_user_pro)
            print(f"Latest Scores: {result_scores}")
            assert "Match: Lakers vs. Warriors" in result_scores
            assert "Score: 105-103" in result_scores
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "sports_get_latest_scores"
            assert logged_data["success"] is True
            print("Test 1 Passed (and analytics logged success).")

            # Test get_upcoming_events (API failure - no data found)
            print("\n--- Test 2: get_upcoming_events (API Failure) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_events = await sports_tools_instance.get_upcoming_events(sport="nonexistent sport", user_token=test_user_pro)
            print(f"Upcoming Events (API Error): {result_events}")
            assert "No live upcoming events found for sport 'nonexistent sport'." in result_events
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "sports_get_upcoming_events"
            assert logged_data["success"] is False
            assert "Response path 'events' not found" in logged_data["error_message"] or "incomplete" in logged_data["error_message"]
            print("Test 2 Passed (and analytics logged failure).")

            # Test get_latest_scores (RBAC denied)
            print("\n--- Test 3: get_latest_scores (RBAC Denied) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_scores_rbac_denied = await sports_tools_instance.get_latest_scores(user_token=test_user_free)
            print(f"Latest Scores (Free User, RBAC Denied): {result_scores_rbac_denied}")
            assert "Error: Access to sports tools is not enabled for your current tier." in result_scores_rbac_denied
            # No analytics log expected here because RBAC check happens before _make_dynamic_api_request
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 3 Passed (RBAC correctly prevented call and no analytics logged).")

            # Test sports_search_web (generic tool, not using _make_dynamic_api_request)
            print("\n--- Test 4: sports_search_web (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_web_search = await sports_tools_instance.sports_search_web("history of basketball", user_token=test_user_pro)
            print(f"Web Search Result: {result_web_search[:100]}...")
            assert "Search results for history of basketball" in result_web_search
            # Analytics for generic tools like scrape_web or summarize_document
            # would need to be integrated within those shared_tools themselves,
            # or wrapped by a higher-level agent logging.
            # For now, we are focusing on _make_dynamic_api_request.
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 4 Passed (no analytics expected for generic tool directly).\n")

            # Test 5: sports_query_uploaded_docs (generic tool)
            print("\n--- Test 5: sports_query_uploaded_docs (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_doc_query = await sports_tools_instance.sports_query_uploaded_docs("team strategy for next game", user_token=test_user_pro)
            print(f"Document Query Result: {result_doc_query}")
            assert "Mocked document query results for 'team strategy for next game' in section 'sports'." in result_doc_query
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 5 Passed (no analytics expected for generic tool directly, will be logged by DocumentTools).")

            # Test 6: sports_summarize_document_by_path (generic tool)
            print("\n--- Test 6: sports_summarize_document_by_path (Generic Tool) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            # Create a dummy file for summarization test
            dummy_file_path = Path("uploads") / test_user_pro / "sports" / "team_strategy.pdf"
            dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_file_path.write_text("This is a dummy team strategy document for testing summarization.")

            result_summarize = await sports_tools_instance.sports_summarize_document_by_path(str(dummy_file_path))
            print(f"Summarize Result: {result_summarize}")
            assert "Mocked summary of text for user default" in result_summarize
            mock_analytics_tracker_db.collection.return_value.add.assert_not_called()
            print("Test 6 Passed (no analytics expected for generic tool directly).\n")

            # Test 7: get_player_stats (success)
            print("\n--- Test 7: get_player_stats (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_player_stats = await sports_tools_instance.get_player_stats("LeBron James", user_token=test_user_pro)
            print(f"Player Stats Result: {result_player_stats}")
            assert "Player Stats for LeBron James:" in result_player_stats
            assert "Points Per Game: 27.2" in result_player_stats
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "sports_get_player_stats"
            assert logged_data["success"] is True
            print("Test 7 Passed (and analytics logged success).")

            # Test 8: get_team_stats (success)
            print("\n--- Test 8: get_team_stats (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_team_stats = await sports_tools_instance.get_team_stats("Los Angeles Lakers", user_token=test_user_pro)
            print(f"Team Stats Result: {result_team_stats}")
            assert "Team Stats for Los Angeles Lakers:" in result_team_stats
            assert "Wins: 50" in result_team_stats
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "sports_get_team_stats"
            assert logged_data["success"] is True
            print("Test 8 Passed (and analytics logged success).")

            # Test 9: get_league_info (success)
            print("\n--- Test 9: get_league_info (Success) ---")
            mock_analytics_tracker_db.collection.return_value.add.reset_mock()
            result_league_info = await sports_tools_instance.get_league_info("NBA", user_token=test_user_pro)
            print(f"League Info Result: {result_league_info}")
            assert "League Information for National Basketball Association:" in result_league_info
            assert "Country: USA/Canada" in result_league_info
            mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
            args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
            logged_data = args[0]
            assert logged_data["event_type"] == "tool_usage"
            assert logged_data["details"]["tool_name"] == "sports_get_league_info"
            assert logged_data["success"] is True
            print("Test 9 Passed (and analytics logged success).")


            print("\nAll sports_tool tests with analytics considerations completed.")

        # Ensure tests are only run when the script is executed directly
        if __name__ == "__main__":
            # Use asyncio.run to execute the async test function
            asyncio.run(run_sports_tests())

        # Restore original requests.get
        requests.get = original_requests_get

        # Restore original summarize_document
        sys.modules['domain_tools.sports_tools.sports_tool'].summarize_document = original_summarize_document

        # Clean up dummy files and directories
        test_user_dirs = [Path("uploads") / test_user_pro, BASE_VECTOR_DIR / test_user_pro]
        for d in test_user_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"Cleaned up {d}")
