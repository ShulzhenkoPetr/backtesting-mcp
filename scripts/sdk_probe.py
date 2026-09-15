from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.types.version import (
    HANDSHAKE_PROTOCOL_VERSIONS,
    LATEST_PROTOCOL_VERSION,
    MODERN_PROTOCOL_VERSIONS,
)

SERVER_SRC = """
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import BaseModel

mcp = MCPServer(name="probe", version="0.0.1")


class Echo(BaseModel):
    said: str


@mcp.tool(description="Echoes a phrase back as structured output.")
async def echo(phrase: str, ctx: Context) -> Echo:
    await ctx.report_progress(1.0, 1.0, "echoed")
    return Echo(said=phrase)


mcp.run()
"""


async def probe_mode(server_file: Path, mode: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(server_file)])
    try:
        async with Client(params, mode=mode) as client:
            tools = await client.list_tools()
            result = await client.call_tool("echo", {"phrase": "hello"})
            print(f"  mode={mode:<9} negotiated={client.protocol_version}")
            print(f"  {'':<15} tools={[t.name for t in tools.tools]}")
            print(f"  {'':<15} output_schema={tools.tools[0].output_schema is not None}")
            print(f"  {'':<15} structured={result.structured_content}")
    except Exception as exc:
        print(f"  mode={mode:<9} FAILED: {type(exc).__name__}: {exc}")


async def probe() -> int:
    server_file = Path(__file__).parent / "_probe_server.py"
    server_file.write_text(SERVER_SRC)
    try:
        print(f"LATEST_PROTOCOL_VERSION     : {LATEST_PROTOCOL_VERSION}")
        print(f"handshake-reachable         : {list(HANDSHAKE_PROTOCOL_VERSIONS)}")
        print(f"modern (stateless envelope) : {list(MODERN_PROTOCOL_VERSIONS)}")
        print()
        for mode in ("auto", "legacy"):
            await probe_mode(server_file, mode)
        return 0
    finally:
        server_file.unlink(missing_ok=True)


def main() -> int:
    return asyncio.run(probe())


if __name__ == "__main__":
    raise SystemExit(main())
