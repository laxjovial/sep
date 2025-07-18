# domain_tools/crypto_tools/crypto_tool.py

import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

# Import the new flexible API request function
from shared_tools.historical_data_tool import make_api_request

# Import user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

# Import UserProfile for type hinting
from backend.models.user_models import UserProfile

logger = logging.getLogger(__name__)

class CryptoTools:
    """
    A collection of tools for cryptocurrency-related operations, including prices,
    historical data, and general information.
    It integrates with external APIs and provides fallback mechanisms.
    """
    def __init__(self, config_manager, firestore_manager, log_event, document_tools):
        self.config_manager = config_manager
        self.firestore_manager = firestore_manager
        self.log_event = log_event
        self.document_tools = document_tools

    @tool
    async def crypto_get_crypto_price(self, crypto_id: str, vs_currencies: str = "usd", user_context: UserProfile = None, provider: str = "coingecko", user_api_keys: list = []) -> str:
        """
        Retrieves the current price of a cryptocurrency.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_crypto_price called for crypto_id: '{crypto_id}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_crypto_price", "success": False, "reason": "RBAC denied"})
            return "Error: Access to crypto tools is not enabled for your current tier."
        
        # --- API Call Logic ---
        params = {"ids": crypto_id, "vs_currencies": vs_currencies}
        api_data = make_api_request(
            provider_name=provider,
            function_name="get_crypto_price",
            params=params,
            user_api_keys=user_api_keys
        )

        if api_data:
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_crypto_price", "success": True, "crypto_id": crypto_id, "vs_currencies": vs_currencies})
            return str(api_data)
        else:
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_crypto_price", "success": False, "crypto_id": crypto_id, "reason": "API data not found"})
            return f"Could not retrieve current price for {crypto_id.upper()}."

    @tool
    async def crypto_get_crypto_id_by_symbol(self, symbol: str, user_context: UserProfile = None, provider: str = "coingecko", user_api_keys: list = []) -> str:
        """
        Retrieves the CoinGecko ID for a given cryptocurrency symbol.
        This is useful when the agent only knows the symbol (e.g., 'BTC') but
        the API requires the CoinGecko ID (e.g., 'bitcoin').
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_crypto_id_by_symbol called for symbol: '{symbol}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_crypto_id_by_symbol", "success": False, "reason": "RBAC denied"})
            return "Error: Access to crypto tools is not enabled for your current tier."

        params = {"query": symbol}
        api_data = make_api_request(
            provider_name=provider,
            function_name="search_crypto_id",
            params=params,
            user_api_keys=user_api_keys
        )

        if api_data and 'coins' in api_data and len(api_data['coins']) > 0:
            # Return the ID of the first matched coin
            crypto_id = api_data['coins'][0]['id']
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_crypto_id_by_symbol", "success": True, "symbol": symbol, "crypto_id": crypto_id})
            return crypto_id
        else:
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_crypto_id_by_symbol", "success": False, "symbol": symbol, "reason": "ID not found"})
            return f"Could not find CoinGecko ID for symbol: {symbol.upper()}."


    @tool
    async def crypto_get_historical_crypto_price(self, crypto_id: str, date: str, vs_currency: str = "usd", user_context: UserProfile = None, provider: str = "coingecko", user_api_keys: list = []) -> str:
        """
        Retrieves the historical price of a cryptocurrency for a specific date.
        Date should be in 'DD-MM-YYYY' format.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_get_historical_crypto_price called for crypto_id: '{crypto_id}' on {date} by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'crypto_tool_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_historical_crypto_price", "success": False, "reason": "RBAC denied"})
            return "Error: Access to crypto tools is not enabled for your current tier."
        
        # --- API Call Logic ---
        params = {"id": crypto_id, "date": date, "vs_currencies": vs_currency}
        api_data = make_api_request(
            provider_name=provider,
            function_name="get_historical_crypto_price",
            params=params,
            user_api_keys=user_api_keys
        )

        if api_data:
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_historical_crypto_price", "success": True, "crypto_id": crypto_id, "date": date, "vs_currency": vs_currency})
            return str(api_data)
        else:
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_get_historical_crypto_price", "success": False, "crypto_id": crypto_id, "date": date, "reason": "API data not found"})
            return f"Could not retrieve historical price for {crypto_id.upper()} on {date}."

    @tool
    async def crypto_search_web(self, query: str, user_context: UserProfile = None, max_chars: int = 2000) -> str:
        """
        Searches the web for cryptocurrency-related information.
        This tool is useful for getting general news, explanations, or current events
        related to cryptocurrencies that are not available via direct API calls.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_search_web called for query: '{query}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'web_search_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_search_web", "success": False, "reason": "RBAC denied"})
            return "Error: Access to web search is not enabled for your current tier."

        # Use the scraper_tool for web search
        # Assuming scraper_tool.scrape_web is available and handles user_context for logging/RBAC
        from shared_tools.scraper_tool import scrape_web # Import locally to avoid circular dependencies if any

        result = await scrape_web(query=query, user_context=user_context, max_chars=max_chars)
        
        # The scrape_web tool should handle its own logging for success/failure
        return result

    @tool
    async def crypto_query_uploaded_docs(self, query: str, user_context: UserProfile = None, section: str = "crypto", k: int = 5, export: bool = False) -> str:
        """
        Queries the user's uploaded and indexed documents specifically related to cryptocurrency
        within a dedicated 'crypto' section to retrieve relevant information using vector similarity search.
        This tool is essential for Retrieval Augmented Generation (RAG) to provide
        answers based on private or specialized knowledge bases (e.g., internal research papers,
        specific project documentation, or detailed whitepapers uploaded by the user).

        Args:
            query (str): The natural language query to search for within the crypto documents.
            user_context (UserProfile): The user's profile object containing user_id, tier, and roles.
            section (str, optional): The specific section within the vector store to query. Defaults to "crypto".
                                     This helps segment user documents by topic.
            k (int, optional): The number of top relevant documents to retrieve. Defaults to 5.
            export (bool, optional): If True, the retrieved document chunks will be exported to a file. Defaults to False.

        Returns:
            str: A string containing the relevant information from the documents, or an error message.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_query_uploaded_docs called for query: '{query}' in section '{section}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'document_query_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_query_uploaded_docs", "success": False, "reason": "RBAC denied"})
            return "Error: Access to document querying is not enabled for your current tier."

        # Leverage the query_uploaded_docs tool from shared_tools
        # Ensure that the query_uploaded_docs tool also handles user_context for RBAC and logging
        from shared_tools.query_uploaded_docs_tool import query_uploaded_docs # Import locally

        result = await query_uploaded_docs(
            query=query,
            user_context=user_context,
            section=section,
            k=k,
            export=export
        )
        # The underlying query_uploaded_docs tool should log its own events
        return result


    @tool
    async def crypto_summarize_document_by_path(self, file_path_str: str, user_context: UserProfile = None) -> str:
        """
        Summarizes a document related to cryptocurrency or blockchain located at the given file path.
        This tool is useful for quickly grasping the main points of long documents.

        Args:
            file_path_str (str): The string representation of the path to the document file.
            user_context (UserProfile): The user's profile object containing user_id, tier, and roles.

        Returns:
            str: A summary of the document, or an error message.
        """
        if user_context is None:
            user_context = UserProfile(user_id="default", username="CLI_User", email="cli@example.com", tier="free", roles=["user"])

        logger.info(f"Tool: crypto_summarize_document_by_path called for path: '{file_path_str}' by user: {user_context.user_id}")

        if not get_user_tier_capability(user_context.user_id, 'document_summarization_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_summarize_document_by_path", "success": False, "reason": "RBAC denied"})
            return "Error: Access to document summarization is not enabled for your current tier."
        
        # Leverage the summarize_document tool from shared_tools.doc_summarizer
        # The DocumentTools instance will have this method if passed during CryptoTools init
        if hasattr(self.document_tools, 'summarize_document'):
            summary = await self.document_tools.summarize_document(
                file_path_str=file_path_str,
                user_context=user_context # Pass user_context for RBAC/logging within summarize_document
            )
            # The summarize_document tool should log its own events
            return summary
        else:
            self.log_event(user_context.user_id, "tool_usage", {"tool_name": "crypto_summarize_document_by_path", "success": False, "file_path": file_path_str, "reason": "Document summarization tool not available"})
            return "Error: Document summarization functionality is not available."

# Example of how you might instantiate and use CryptoTools if not part of a larger framework:
# This part is typically for testing or direct usage, not for the agent framework itself.
import asyncio
import requests
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
import shutil

# Mock dependencies for isolated testing
class MockConfigManager:
    def get_api_provider_config(self, domain, provider_name):
        if domain == "crypto" and provider_name == "coingecko":
            return {"base_url": "https://api.coingecko.com/api/v3", "functions": {
                "get_crypto_price": {"endpoint": "/simple/price", "params": ["ids", "vs_currencies"]},
                "search_crypto_id": {"endpoint": "/search", "params": ["query"]},
                "get_historical_crypto_price": {"endpoint": "/coins/{id}/history", "params": ["id", "date", "vs_currencies"]}
            }}
        return None

# Mock the UserProfile class if not fully available during testing
class MockUserProfile:
    def __init__(self, user_id, tier, roles):
        self.user_id = user_id
        self.username = f"User {user_id}"
        self.email = f"{user_id}@example.com"
        self.tier = tier
        self.roles = roles

class MockAnalyticsTrackerDB:
    def collection(self, name):
        return MagicMock()

# Mock log_event
mock_analytics_tracker_db = MockAnalyticsTrackerDB()
def mock_log_event(user_id, event_type, details):
    logger.info(f"Mock Log Event - User: {user_id}, Type: {event_type}, Details: {details}")
    mock_analytics_tracker_db.collection("analytics_events").add({"user_id": user_id, "event_type": event_type, "details": details})


# Mock get_user_tier_capability for testing RBAC
original_get_user_tier_capability = get_user_tier_capability
def mock_get_user_tier_capability(user_id: str, capability: str, default: Any = None, user_tier: Optional[str] = None, user_roles: Optional[List[str]] = None) -> Any:
    # This mock determines access for testing purposes
    if capability in ['finance_tool_access', 'crypto_tool_access', 'web_search_access', 'document_query_access', 'document_summarization_access']:
        if user_tier in ["pro", "premium", "admin"]:
            return True
        elif capability == 'web_search_access' and user_tier == 'free': # Free tier gets some web search
            return True
        return False
    return default

# Patch the actual get_user_tier_capability during testing
import sys
sys.modules['utils.user_manager'].get_user_tier_capability = mock_get_user_tier_capability

# Mock DocumentTools (as passed into CryptoTools)
class MockDocumentTools:
    def __init__(self):
        # Mock load_vectorstore to return a mock retriever
        self.load_vectorstore = AsyncMock(return_value=MagicMock(as_retriever=MagicMock(return_value=AsyncMock())))
        # Mock query_uploaded_docs results
        self.query_uploaded_docs = AsyncMock(return_value="Mocked relevant crypto document chunks for query.")
        # Mock summarize_document results
        self.summarize_document = AsyncMock(return_value="Mocked summary of document at path.")

# Create a mock for shared_tools.query_uploaded_docs_tool.query_uploaded_docs
# Since crypto_tool directly imports query_uploaded_docs, we need a module-level mock
# This mock will be called by crypto_query_uploaded_docs directly
async def mock_query_uploaded_docs_global(
    query: str,
    user_context: Any,
    section: str,
    k: int,
    export: bool
) -> str:
    logger.info(f"GLOBAL Mock query_uploaded_docs called for query: '{query}' in section '{section}' by user: {user_context.user_id}")
    # Simulate RBAC check within the mock if needed, or assume it's handled upstream
    if not mock_get_user_tier_capability(user_context.user_id, 'document_query_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        return "Error: Access to document querying is not enabled for your current tier (mocked)."
    # Simulate export
    if export:
        export_path = Path(f"/tmp/{user_context.user_id}/mock_export.txt")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(f"Exported content for query: {query}")
        return f"Mocked relevant crypto document chunks for query. Results exported to: `{export_path}`"
    return f"Mocked relevant crypto document chunks for query: '{query}' from section '{section}'."

# Patch the global function for testing purposes
sys.modules['shared_tools.query_uploaded_docs_tool'].query_uploaded_docs = mock_query_uploaded_docs_global

# Create a mock for shared_tools.scraper_tool.scrape_web
async def mock_scrape_web_global(query: str, user_context: Any, max_chars: int) -> str:
    logger.info(f"GLOBAL Mock scrape_web called for query: '{query}' by user: {user_context.user_id}")
    if not mock_get_user_tier_capability(user_context.user_id, 'web_search_access', False, user_tier=user_context.tier, user_roles=user_context.roles):
        return "Error: Access to web search is not enabled for your current tier (mocked)."
    mock_content = f"Mocked web search results for '{query}'. This is a simulated response up to {max_chars} characters. " * 5
    return mock_content[:max_chars]

# Patch the global function for testing purposes
sys.modules['shared_tools.scrapper_tool'].scrape_web = mock_scrape_web_global

# Create a mock for shared_tools.doc_summarizer.summarize_document
# Note: The CryptoTools class calls self.document_tools.summarize_document,
# so we need to ensure MockDocumentTools has this mock.
# No direct module-level patch needed here for crypto_tool itself,
# but rather for the instance passed to CryptoTools.

async def run_crypto_tests(crypto_tools_instance: CryptoTools):
    print("--- Running CryptoTools tests ---")

    mock_user_free_profile = MockUserProfile(user_id="test_user_free", tier="free", roles=["user"])
    mock_user_pro_profile = MockUserProfile(user_id="test_user_pro", tier="pro", roles=["user", "pro"])

    # Temporarily patch requests.get and requests.post for API call simulation
    original_requests_get = requests.get
    # requests.post = MagicMock() # If you use post requests, mock it too

    # Test 1: Get crypto price (success)
    print("\n--- Test 1: Get crypto price (success) ---")
    requests.get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: {"bitcoin": {"usd": 60000}}))
    result1 = await crypto_tools_instance.crypto_get_crypto_price("bitcoin", user_context=mock_user_pro_profile)
    print(f"Result 1 (Crypto Price): {result1}")
    assert "60000" in result1
    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
    logged_data = args[0]
    assert logged_data["event_type"] == "tool_usage"
    assert logged_data["details"]["tool_name"] == "crypto_get_crypto_price"
    assert logged_data["success"] is True
    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
    print("Test 1 Passed.")

    # Test 2: Get crypto price (failure - API data not found)
    print("\n--- Test 2: Get crypto price (failure - API data not found) ---")
    requests.get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: {}))
    result2 = await crypto_tools_instance.crypto_get_crypto_price("nonexistentcoin", user_context=mock_user_pro_profile)
    print(f"Result 2 (Crypto Price Failure): {result2}")
    assert "Could not retrieve current price" in result2
    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
    logged_data = args[0]
    assert logged_data["event_type"] == "tool_usage"
    assert logged_data["details"]["tool_name"] == "crypto_get_crypto_price"
    assert logged_data["success"] is False
    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
    print("Test 2 Passed.")

    # Test 3: Get CoinGecko ID (success)
    print("\n--- Test 3: Get CoinGecko ID (success) ---")
    requests.get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: {"coins": [{"id": "ethereum", "name": "Ethereum", "symbol": "ETH"}]}))
    result3 = await crypto_tools_instance.crypto_get_crypto_id_by_symbol("ETH", user_context=mock_user_pro_profile)
    print(f"Result 3 (CoinGecko ID): {result3}")
    assert "ethereum" == result3
    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
    logged_data = args[0]
    assert logged_data["event_type"] == "tool_usage"
    assert logged_data["details"]["tool_name"] == "crypto_get_crypto_id_by_symbol"
    assert logged_data["success"] is True
    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
    print("Test 3 Passed.")

    # Test 4: Get historical crypto price (success)
    print("\n--- Test 4: Get historical crypto price (success) ---")
    requests.get = MagicMock(return_value=MagicMock(status_code=200, json=lambda: {"market_data": {"current_price": {"usd": 50000}}}))
    result4 = await crypto_tools_instance.crypto_get_historical_crypto_price("bitcoin", "01-01-2023", user_context=mock_user_pro_profile)
    print(f"Result 4 (Historical Price): {result4}")
    assert "50000" in result4
    mock_analytics_tracker_db.collection.return_value.add.assert_called_once()
    args, kwargs = mock_analytics_tracker_db.collection.return_value.add.call_args
    logged_data = args[0]
    assert logged_data["event_type"] == "tool_usage"
    assert logged_data["details"]["tool_name"] == "crypto_get_historical_crypto_price"
    assert logged_data["success"] is True
    mock_analytics_tracker_db.collection.return_value.add.reset_mock()
    print("Test 4 Passed.")

    # Test 5: Web Search (success and RBAC check)
    print("\n--- Test 5: Web Search (success and RBAC check) ---")
    # Pro user has web search access (mocked)
    result5_pro = await crypto_tools_instance.crypto_search_web("What is Bitcoin halving?", user_context=mock_user_pro_profile)
    print(f"Result 5 (Web Search Pro): {result5_pro}")
    assert "Mocked web search results for 'What is Bitcoin halving?'" in result5_pro
    # No analytics tracker assert here because scrape_web handles its own logging.

    # Free user also has web search access (mocked for this capability)
    result5_free = await crypto_tools_instance.crypto_search_web("Latest Ethereum news", user_context=mock_user_free_profile)
    print(f"Result 5 (Web Search Free): {result5_free}")
    assert "Mocked web search results for 'Latest Ethereum news'" in result5_free
    print("Test 5 Passed.")


    # Test 6: Query Uploaded Docs (success and RBAC check)
    print("\n--- Test 6: Query Uploaded Docs (success and RBAC check) ---")
    # Pro user has document query access (mocked)
    result6_pro = await crypto_tools_instance.crypto_query_uploaded_docs(
        "Explain Proof of Stake", user_context=mock_user_pro_profile, section="crypto", export=True
    )
    print(f"Result 6 (Query Docs Pro): {result6_pro}")
    assert "Mocked relevant crypto document chunks for query: 'Explain Proof of Stake'" in result6_pro
    assert Path(f"/tmp/{mock_user_pro_profile.user_id}/mock_export.txt").exists()
    Path(f"/tmp/{mock_user_pro_profile.user_id}/mock_export.txt").unlink() # Clean up

    # Free user does NOT have document query access (mocked)
    result6_free = await crypto_tools_instance.crypto_query_uploaded_docs(
        "DeFi protocols", user_context=mock_user_free_profile, section="crypto"
    )
    print(f"Result 6 (Query Docs Free): {result6_free}")
    assert "Error: Access to document querying is not enabled for your current tier (mocked)." in result6_free
    print("Test 6 Passed.")


    # Test 7: Summarize Document by Path (success and RBAC check)
    print("\n--- Test 7: Summarize Document by Path (success and RBAC check) ---")
    # Create a dummy file for summarization
    dummy_file_path = Path("uploads") / mock_user_pro_profile.user_id / "dummy_file.txt"
    dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_file_path.write_text("This is a test document about cryptocurrency. It discusses blockchain technology.")

    # Pro user has document summarization access (mocked)
    mock_document_tools_instance.summarize_document.return_value = "Mocked summary of dummy_file.txt"
    result_summarize = await crypto_tools_instance.crypto_summarize_document_by_path(str(dummy_file_path), user_context=mock_user_pro_profile)
    print(f"Summarize Result: {result_summarize}")
    assert "Mocked summary of dummy_file.txt" in result_summarize # Check for mock summary from DocumentTools
    # The logging for this should be handled by the summarize_document tool itself,
    # but for this specific test, we can check the analytics tracker if the tool
    # *itself* uses it before delegating, or if the mock for summarize_document logs.
    # In this setup, mock_document_tools_instance.summarize_document is directly called,
    # and it doesn't log to mock_analytics_tracker_db directly, so skip that assert.
    print("Test 7 Passed.")

    print("\nAll crypto_tool tests with live API simulation and analytics considerations completed.")

    # Ensure tests are only run when the script is executed directly
if __name__ == "__main__":
    # Instantiate CryptoTools with mocks
    mock_config_manager = MockConfigManager()
    mock_firestore_manager = MagicMock() # Not directly used in CryptoTools methods, but required for init
    mock_document_tools_instance = MockDocumentTools() # Pass a mock instance

    crypto_tools_instance = CryptoTools(
        config_manager=mock_config_manager,
        firestore_manager=mock_firestore_manager,
        log_event=mock_log_event, # Pass the mock log_event function
        document_tools=mock_document_tools_instance
    )
    asyncio.run(run_crypto_tests(crypto_tools_instance))

    # Restore original requests.get
    requests.get = original_requests_get
    # if 'original_requests_post' in locals():
    #     requests.post = original_requests_post

    # Restore original get_user_tier_capability
    sys.modules['utils.user_manager'].get_user_tier_capability = original_get_user_tier_capability

    # Clean up dummy files and directories
    test_user_dirs = [Path("uploads") / "test_user_pro", Path("vector_stores") / "test_user_pro", Path("/tmp/test_user_pro")]
    for d in test_user_dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"Cleaned up directory: {d}")