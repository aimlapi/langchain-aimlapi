# AIMLAPI_HEADERS: sent with each request to let the server know who you are
# Helps with analytics, debugging, and enforcing usage policies
AIMLAPI_HEADERS = {
    # Tells the API which application is making the call
    "HTTP-Referer": "https://github.com/langchain-ai/langchain",
    # Identifies the client or library name for tracking
    "X-Title": "LangChain",
    # Rebate attribution id (part_...) for the "langchain" partner row in
    # AI/ML API's rebate_partners table. Do not repoint this to a different
    # partner without also updating the backend record.
    "X-AIMLAPI-Partner-ID": "part_18CxLUehxepR5hLOIyrIXbv0",
    "X-AIMLAPI-Source": "agent/langchain",
}
