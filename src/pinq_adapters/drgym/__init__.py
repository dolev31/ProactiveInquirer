"""DeepResearchGym adapter: hosted open-web search, Researchy queries, `.q`/`.a` reports.

The only open-corpus suite here. `suite.DrGymSuite(root, offline=True)` replays a
content-addressed search cache and needs no API key; `offline=False` needs DRGYM_API_KEY and
records what it fetches into the same cache.
"""
