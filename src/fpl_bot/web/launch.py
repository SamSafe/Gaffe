"""Service entry point; the browser launcher uses the same app factory."""

import uvicorn

from fpl_bot.web.app import create_app


def main():
    uvicorn.run(create_app(), host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
