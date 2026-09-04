from typing import Dict, Mapping, Optional

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


def merge_aimlapi_headers(
    headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Overlay caller headers on a fresh copy of ``AIMLAPI_HEADERS``.

    Merge, never assign. ``default_headers`` is a pydantic field with a
    ``default_factory``, and a ``default_factory`` fires only when the field is
    *not supplied*. So a caller who passed one header of their own -- a trace
    id, a proxy tag, anything -- used to replace the dict wholesale and send
    none of the four attribution headers. The request still succeeded, so
    neither side could see it happen: the call simply became untracked spend.

    An explicit value for one of our own keys still wins. That is deliberate --
    a reseller or a downstream framework may need to identify itself -- but it
    now takes naming the key, not merely adding an unrelated one.

    Returns a new dict every time, so no caller shares mutable state with the
    module-level constant.
    """
    merged = dict(AIMLAPI_HEADERS)
    if headers:
        merged.update(headers)
    return merged
