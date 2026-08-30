import asyncio
import selectors

import uvicorn


async def main():
    config = uvicorn.Config(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
    )

    server = uvicorn.Server(config)

    await server.serve()


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )
