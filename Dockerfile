# Builds the magpie-search MCP server (for Glama checks / container use).
# The server speaks the Model Context Protocol over stdio (JSON-RPC 2.0) and
# responds to introspection (initialize + tools/list) without needing an index
# or the embedding model (those load lazily only on semantic search).
FROM python:3.12-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .

# stdio MCP server entrypoint
ENTRYPOINT ["magpie-search-mcp"]
